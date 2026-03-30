"""
adapter.py — Code extraction and adaptation for train.py integration.

Takes scanned-and-approved source code, extracts a specific component
(class or function group), transforms it to fit train.py's structure,
and produces an insertion-ready code string.

The adapter NEVER executes code. It produces text.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Top-level names in train.py that must not be shadowed
TRAIN_PY_NAMES = {
    "GPTConfig", "GPT", "CausalSelfAttention", "MLP", "Block",
    "MuonAdamW", "norm", "has_ve", "apply_rotary_emb",
    "adamw_step_fused", "muon_step_fused", "polar_express_coeffs",
    "build_model_config", "get_lr_multiplier", "get_muon_momentum",
    "get_weight_decay",
}

# Zone insertion points (line numbers in baseline train.py at commit 649e8c6)
ZONE_INSERTION = {
    "model_architecture": {
        "description": "After Block class, before GPT class",
        "after_marker": "class Block(nn.Module):",
        "before_marker": "class GPT(nn.Module):",
    },
    "optimizer": {
        "description": "After MuonAdamW, before hyperparameters",
        "after_marker": "class MuonAdamW(torch.optim.Optimizer):",
        "before_marker": "# Hyperparameters",
    },
    "schedules": {
        "description": "After schedule functions, before training loop",
        "after_marker": "def get_weight_decay(progress):",
        "before_marker": "# Training loop",
    },
}

# Module allowlist (from governance_v2.json, duplicated here for offline use)
ALLOWED_MODULES = {
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "torch.cuda", "torch.amp",
    "math", "os", "sys", "time", "typing", "dataclasses",
    "functools", "itertools", "collections", "abc", "copy",
    "warnings", "contextlib", "pathlib", "re", "string",
    "struct", "io", "enum", "gc", "platform", "numpy",
}


@dataclass
class ComponentInfo:
    """Metadata about a top-level component in a source file."""
    name: str
    kind: str  # "class", "function", "assignment"
    base_classes: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    dependencies: list[str] = field(default_factory=list)
    imports_needed: list[str] = field(default_factory=list)


@dataclass
class DependencyConflict:
    """An import that can't be resolved."""
    name: str
    reason: str
    severity: str  # "blocking" or "missing"


@dataclass
class AdaptationResult:
    """Result of adapting a component for train.py."""
    source_repo: str
    source_file: str
    extracted_component: str
    all_extracted_names: list[str]
    adapted_code: str
    target_zone: str
    renames_applied: dict[str, str]
    dependency_conflicts: list[DependencyConflict]
    line_count: int
    can_proceed: bool
    reason: str = ""


def build_component_map(source: str) -> dict[str, ComponentInfo]:
    """Parse source and return all top-level names with metadata."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    lines = source.splitlines()
    components = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            components[node.name] = ComponentInfo(
                name=node.name,
                kind="class",
                base_classes=bases,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            )
        elif isinstance(node, ast.FunctionDef):
            components[node.name] = ComponentInfo(
                name=node.name,
                kind="function",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    components[target.id] = ComponentInfo(
                        name=target.id,
                        kind="assignment",
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                    )

    # Trace dependencies: names referenced inside each component
    for name, info in components.items():
        start = info.line_start - 1
        end = info.line_end
        chunk = "\n".join(lines[start:end])
        try:
            chunk_tree = ast.parse(chunk)
        except SyntaxError:
            continue
        referenced = set()
        for node in ast.walk(chunk_tree):
            if isinstance(node, ast.Name) and node.id in components and node.id != name:
                referenced.add(node.id)
        info.dependencies = sorted(referenced)

    return components


def extract_component(
    source: str,
    target_name: str,
    include_dependencies: bool = True,
    max_depth: int = 3,
) -> tuple[str, list[str], list[str]]:
    """Extract a component and its dependencies from source.

    Returns (extracted_code, required_imports, all_extracted_names).
    """
    components = build_component_map(source)
    if target_name not in components:
        return "", [], []

    # Gather the target + dependencies
    to_extract = {target_name}
    if include_dependencies:
        frontier = {target_name}
        for _ in range(max_depth):
            next_frontier = set()
            for name in frontier:
                if name in components:
                    for dep in components[name].dependencies:
                        if dep not in to_extract:
                            to_extract.add(dep)
                            next_frontier.add(dep)
            frontier = next_frontier
            if not frontier:
                break

    # Sort by line number for stable output
    ordered = sorted(to_extract, key=lambda n: components[n].line_start if n in components else 0)

    # Extract code lines
    lines = source.splitlines()
    code_chunks = []
    for name in ordered:
        if name not in components:
            continue
        info = components[name]
        chunk = "\n".join(lines[info.line_start - 1 : info.line_end])
        code_chunks.append(chunk)

    # Collect imports from the source file
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n\n".join(code_chunks), [], list(ordered)

    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_line = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            imports.append(import_line)

    return "\n\n\n".join(code_chunks), imports, list(ordered)


def resolve_imports(imports: list[str]) -> tuple[list[str], list[DependencyConflict]]:
    """Split imports into allowed and conflicting."""
    kept = []
    conflicts = []

    for imp_line in imports:
        try:
            tree = ast.parse(imp_line)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in ALLOWED_MODULES or alias.name in ALLOWED_MODULES:
                        kept.append(imp_line)
                    else:
                        conflicts.append(DependencyConflict(
                            name=alias.name,
                            reason=f"Module '{module_root}' not in allowlist",
                            severity="blocking",
                        ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in ALLOWED_MODULES or node.module in ALLOWED_MODULES:
                        kept.append(imp_line)
                    else:
                        conflicts.append(DependencyConflict(
                            name=node.module,
                            reason=f"Module '{module_root}' not in allowlist",
                            severity="blocking",
                        ))

    # Deduplicate
    kept = list(dict.fromkeys(kept))
    return kept, conflicts


def detect_collisions(names: list[str]) -> dict[str, str]:
    """Check for name collisions with train.py. Returns {original: renamed}."""
    renames = {}
    for name in names:
        if name in TRAIN_PY_NAMES:
            renames[name] = f"{name}_imported"
    return renames


def apply_renames(code: str, renames: dict[str, str]) -> str:
    """Apply name renames throughout the code string."""
    for old_name, new_name in renames.items():
        # Use word-boundary replacement to avoid partial matches
        code = re.sub(rf'\b{re.escape(old_name)}\b', new_name, code)
    return code


def adapt(
    source: str,
    file_path: str,
    repo_full_name: str,
    target_component: str,
    target_zone: str,
) -> AdaptationResult:
    """Main entry point: extract, transform, and prepare code for train.py.

    Does NOT insert code or run safety_scanner — caller handles those.
    """
    # Extract component and dependencies
    code, imports, all_names = extract_component(source, target_component)

    if not code:
        return AdaptationResult(
            source_repo=repo_full_name,
            source_file=file_path,
            extracted_component=target_component,
            all_extracted_names=[],
            adapted_code="",
            target_zone=target_zone,
            renames_applied={},
            dependency_conflicts=[],
            line_count=0,
            can_proceed=False,
            reason=f"Component '{target_component}' not found in source",
        )

    # Resolve imports
    kept_imports, conflicts = resolve_imports(imports)

    # Detect and apply name collision renames
    renames = detect_collisions(all_names)
    if renames:
        code = apply_renames(code, renames)

    # Check line count limit
    line_count = len(code.splitlines())
    if line_count > 200:
        return AdaptationResult(
            source_repo=repo_full_name,
            source_file=file_path,
            extracted_component=target_component,
            all_extracted_names=all_names,
            adapted_code=code,
            target_zone=target_zone,
            renames_applied=renames,
            dependency_conflicts=conflicts,
            line_count=line_count,
            can_proceed=False,
            reason=f"Extracted code is {line_count} lines (max 200)",
        )

    # Check for blocking dependency conflicts
    blocking = [c for c in conflicts if c.severity == "blocking"]
    if blocking:
        names = ", ".join(c.name for c in blocking)
        return AdaptationResult(
            source_repo=repo_full_name,
            source_file=file_path,
            extracted_component=target_component,
            all_extracted_names=all_names,
            adapted_code=code,
            target_zone=target_zone,
            renames_applied=renames,
            dependency_conflicts=conflicts,
            line_count=line_count,
            can_proceed=False,
            reason=f"Blocked dependencies: {names}",
        )

    # Build the final adapted code with imports
    import_block = "\n".join(kept_imports)
    if import_block:
        adapted = f"{import_block}\n\n{code}"
    else:
        adapted = code

    return AdaptationResult(
        source_repo=repo_full_name,
        source_file=file_path,
        extracted_component=target_component,
        all_extracted_names=all_names,
        adapted_code=adapted,
        target_zone=target_zone,
        renames_applied=renames,
        dependency_conflicts=conflicts,
        line_count=line_count,
        can_proceed=True,
    )
