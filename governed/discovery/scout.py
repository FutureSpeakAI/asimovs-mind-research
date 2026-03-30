"""
scout.py — Asimov's Mind GitHub Discovery Scout

Searches GitHub for ML code (optimizers, attention mechanisms, training tricks,
normalization methods) and ranks results by relevance and trust.

Uses the GitHub REST API with optional token authentication and respects rate
limits via the X-RateLimit-Remaining header.

Usage:
    from governed.discovery.scout import run_scout
    report = run_scout()                        # all strategies
    report = run_scout(strategies=["optimizer_search"], min_relevance=0.5)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
MAX_FILE_SIZE = 100 * 1024  # 100 KB cap for fetched file content

# Back-off when rate-limited (seconds)
RATE_LIMIT_PAUSE = 60

# ---------------------------------------------------------------------------
# Search strategy vocabularies
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, list[str]] = {
    "optimizer_search": [
        "muon",
        "soap",
        "schedule-free",
        "came",
        "ademamix",
        "grafting",
    ],
    "attention_search": [
        "linear attention",
        "sparse attention",
        "differential attention",
        "flash attention",
        "multi-latent",
    ],
    "training_trick_search": [
        "z-loss",
        "logit capping",
        "auxiliary loss",
        "token dropping",
    ],
    "normalization_search": [
        "rms norm",
        "qk norm",
        "deep norm",
        "sandwich norm",
    ],
}

# ---------------------------------------------------------------------------
# Known-author whitelist (lowercased for comparison)
# ---------------------------------------------------------------------------

KNOWN_AUTHORS: set[str] = {
    "karpathy",
    "kellerjordan",
    "tysam-code",
    "pytorch-labs",
    "google-deepmind",
    "eleutherai",
    "huggingface",
    "youjiacheng",
}

# ---------------------------------------------------------------------------
# License classification
# ---------------------------------------------------------------------------

_PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}
_COPYLEFT_LICENSES = {"gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0"}


def _license_score(spdx: Optional[str]) -> float:
    """Return a trust sub-score for the licence.  No licence = hard block (0.0)."""
    if not spdx:
        return 0.0
    key = spdx.lower()
    if key in _PERMISSIVE_LICENSES:
        return 1.0
    if key in _COPYLEFT_LICENSES:
        return 0.6
    # Unknown but present — treat as moderate
    return 0.4


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single scored candidate repository."""

    repo_full_name: str
    repo_url: str
    description: str
    license_spdx: Optional[str]
    stars: int
    forks: int
    relevance_score: float
    trust_score: float
    target_zone: str  # which strategy found it
    query_terms_matched: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SearchResult({self.repo_full_name!r}, "
            f"rel={self.relevance_score:.2f}, "
            f"trust={self.trust_score:.2f})"
        )


@dataclass
class ScoutReport:
    """Aggregated output of a scouting run."""

    timestamp: str
    candidates: list[SearchResult]  # sorted by relevance DESC
    filtered_count: int  # how many were dropped by thresholds

    def summary(self) -> str:
        return (
            f"ScoutReport @ {self.timestamp}: "
            f"{len(self.candidates)} candidates, "
            f"{self.filtered_count} filtered out"
        )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class _RateLimited(Exception):
    """Raised when GitHub returns 403 due to rate limiting."""


def _session() -> requests.Session:
    """Build a session with optional auth token."""
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    s.headers["X-GitHub-Api-Version"] = "2022-11-28"
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _check_rate_limit(resp: requests.Response) -> None:
    """Inspect rate-limit headers and pause if necessary."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) <= 1:
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), RATE_LIMIT_PAUSE)
        logger.warning("Rate limit nearly exhausted — sleeping %d s", wait)
        time.sleep(wait)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), RATE_LIMIT_PAUSE)
        logger.warning("Rate-limited by GitHub — sleeping %d s", wait)
        time.sleep(wait)
        raise _RateLimited(f"Rate-limited; waited {wait}s")


# ---------------------------------------------------------------------------
# GitHub API wrappers
# ---------------------------------------------------------------------------

def search_github(
    query: str,
    sort: str = "stars",
    max_results: int = 30,
) -> list[dict]:
    """Search GitHub repositories and return raw API items.

    Parameters
    ----------
    query : str
        The search query (GitHub search syntax).
    sort : str
        Sort field — "stars", "forks", or "updated".
    max_results : int
        Cap on number of items returned (API page size is 100 max).

    Returns
    -------
    list[dict]
        Raw repository objects from the GitHub Search API.
    """
    sess = _session()
    per_page = min(max_results, 100)
    params = {
        "q": query,
        "sort": sort,
        "order": "desc",
        "per_page": per_page,
    }

    items: list[dict] = []
    page = 1

    while len(items) < max_results:
        params["page"] = page
        try:
            resp = sess.get(f"{GITHUB_API}/search/repositories", params=params, timeout=15)
            _check_rate_limit(resp)
            resp.raise_for_status()
        except _RateLimited:
            # After sleeping, retry once
            try:
                resp = sess.get(f"{GITHUB_API}/search/repositories", params=params, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.error("GitHub search failed after rate-limit retry: %s", exc)
                break
        except requests.RequestException as exc:
            logger.error("GitHub search request failed: %s", exc)
            break

        data = resp.json()
        batch = data.get("items", [])
        if not batch:
            break

        items.extend(batch)
        # GitHub caps search results at 1000
        if len(batch) < per_page or data.get("total_count", 0) <= len(items):
            break
        page += 1

    return items[:max_results]


def fetch_file_listing(
    repo_full_name: str,
    branch: str = "main",
) -> list[str]:
    """Return a list of ``.py`` file paths in *repo_full_name* using the Git
    tree API (recursive).

    Parameters
    ----------
    repo_full_name : str
        Owner/repo, e.g. ``"karpathy/nanoGPT"``.
    branch : str
        Branch or tag to list.

    Returns
    -------
    list[str]
        Paths relative to the repo root, filtered to ``.py`` files.
    """
    sess = _session()
    url = f"{GITHUB_API}/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    try:
        resp = sess.get(url, timeout=15)
        _check_rate_limit(resp)
        resp.raise_for_status()
    except _RateLimited:
        try:
            resp = sess.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Tree listing failed after rate-limit retry: %s", exc)
            return []
    except requests.RequestException as exc:
        logger.error("Tree listing failed for %s: %s", repo_full_name, exc)
        return []

    tree = resp.json().get("tree", [])
    return [
        node["path"]
        for node in tree
        if node.get("type") == "blob" and node["path"].endswith(".py")
    ]


def fetch_file_content(
    repo_full_name: str,
    path: str,
    branch: str = "main",
) -> str:
    """Download a single file's raw content from GitHub.

    Parameters
    ----------
    repo_full_name : str
        Owner/repo.
    path : str
        File path relative to repo root.
    branch : str
        Branch or tag.

    Returns
    -------
    str
        The file content (decoded as UTF-8).  Truncated to ``MAX_FILE_SIZE``.
    """
    sess = _session()
    url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
    try:
        resp = sess.get(url, timeout=15, stream=True)
        _check_rate_limit(resp)
        resp.raise_for_status()
    except _RateLimited:
        try:
            resp = sess.get(url, timeout=15, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("File fetch failed after rate-limit retry: %s", exc)
            return ""
    except requests.RequestException as exc:
        logger.error("File fetch failed for %s/%s: %s", repo_full_name, path, exc)
        return ""

    content = resp.content[:MAX_FILE_SIZE]
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _topic_match_score(repo: dict, vocab: list[str]) -> tuple[float, list[str]]:
    """Fraction of vocab terms found in repo description + topics.

    Returns (score, matched_terms).
    """
    text = " ".join([
        (repo.get("description") or ""),
        " ".join(repo.get("topics") or []),
        (repo.get("full_name") or ""),
    ]).lower()

    matched = [term for term in vocab if term.lower() in text]
    if not vocab:
        return 0.0, []
    return len(matched) / len(vocab), matched


def _recency_score(repo: dict) -> float:
    """Score based on last push date.  Repos > 3 years old get 0."""
    pushed_at = repo.get("pushed_at")
    if not pushed_at:
        return 0.0

    try:
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0

    now = datetime.now(timezone.utc)
    age_days = (now - pushed).days
    max_age = 3 * 365  # 3 years

    if age_days >= max_age:
        return 0.0
    # Linear decay
    return 1.0 - (age_days / max_age)


def _signal_score(repo: dict) -> float:
    """Log-scaled star/fork signal.  Forks count as half a star."""
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    effective = stars + 0.5 * forks

    if effective <= 0:
        return 0.0
    # log10(1) = 0, log10(10) = 1, log10(100_000) ~= 5
    # Normalise so 10k effective stars ~ 1.0
    return min(math.log10(1 + effective) / math.log10(10_001), 1.0)


def _specificity_score(repo: dict) -> float:
    """Smaller repos (fewer files) are more likely to be focused research code.

    Uses repo "size" (KB) as a proxy since the search API doesn't return file
    counts.  < 500 KB = 1.0, > 50 MB = 0.0.
    """
    size_kb = repo.get("size", 0)
    if size_kb <= 500:
        return 1.0
    if size_kb >= 50_000:
        return 0.0
    # Linear interpolation
    return 1.0 - (size_kb - 500) / (50_000 - 500)


def _known_author_score(repo: dict) -> float:
    """1.0 if the owner is on the whitelist, 0.0 otherwise."""
    owner = (repo.get("owner") or {}).get("login", "")
    return 1.0 if owner.lower() in KNOWN_AUTHORS else 0.0


def _age_trust_score(repo: dict) -> float:
    """Repos < 30 days old get 0 (too new to trust)."""
    created_at = repo.get("created_at")
    if not created_at:
        return 0.0

    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0

    age_days = (datetime.now(timezone.utc) - created).days
    if age_days < 30:
        return 0.0
    # Ramp up over 6 months
    return min(age_days / 180, 1.0)


def _contributor_score(repo: dict) -> float:
    """Rough contributor signal from forks + watchers as proxy.

    The search API doesn't return contributor count directly.  We use
    ``watchers_count > 1`` as a reasonable proxy for multi-person projects.
    """
    watchers = repo.get("watchers_count", 0)
    forks = repo.get("forks_count", 0)
    # If there are forks or multiple watchers, likely >1 contributor
    if forks > 0 or watchers > 1:
        return 1.0
    return 0.3


def _activity_score(repo: dict) -> float:
    """Score based on open issues and watchers — signs of active maintenance."""
    issues = repo.get("open_issues_count", 0)
    watchers = repo.get("watchers_count", 0)

    # Some open issues = maintained; too many = maybe abandoned bug tracker
    issue_signal = min(issues / 20, 1.0) if issues <= 100 else 0.5
    watcher_signal = min(watchers / 50, 1.0)

    return 0.5 * issue_signal + 0.5 * watcher_signal


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_candidate(repo_data: dict, strategy: str) -> SearchResult:
    """Score a single repository against a search strategy.

    Parameters
    ----------
    repo_data : dict
        Raw repository dict from the GitHub Search API.
    strategy : str
        Key into ``STRATEGIES`` (e.g. ``"optimizer_search"``).

    Returns
    -------
    SearchResult
        Fully scored candidate.
    """
    vocab = STRATEGIES.get(strategy, [])

    # --- Relevance (0-1) ---
    topic, matched_terms = _topic_match_score(repo_data, vocab)
    recency = _recency_score(repo_data)
    signal = _signal_score(repo_data)
    specificity = _specificity_score(repo_data)
    author = _known_author_score(repo_data)

    relevance = (
        0.30 * topic
        + 0.20 * recency
        + 0.20 * signal
        + 0.15 * specificity
        + 0.15 * author
    )

    # --- Trust (0-1) ---
    lic_spdx = None
    lic_info = repo_data.get("license")
    if isinstance(lic_info, dict):
        lic_spdx = lic_info.get("spdx_id")

    lic = _license_score(lic_spdx)
    age = _age_trust_score(repo_data)
    contribs = _contributor_score(repo_data)
    activity = _activity_score(repo_data)

    trust = (
        0.35 * lic
        + 0.25 * age
        + 0.20 * contribs
        + 0.20 * activity
    )

    return SearchResult(
        repo_full_name=repo_data.get("full_name", ""),
        repo_url=repo_data.get("html_url", ""),
        description=(repo_data.get("description") or "")[:200],
        license_spdx=lic_spdx,
        stars=repo_data.get("stargazers_count", 0),
        forks=repo_data.get("forks_count", 0),
        relevance_score=round(relevance, 4),
        trust_score=round(trust, 4),
        target_zone=strategy,
        query_terms_matched=matched_terms,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_scout(
    strategies: Optional[list[str]] = None,
    min_relevance: float = 0.35,
    min_trust: float = 0.5,
) -> ScoutReport:
    """Run a full scouting pass across one or more strategies.

    Parameters
    ----------
    strategies : list[str] | None
        Which strategies to execute.  ``None`` = all of them.
    min_relevance : float
        Minimum relevance score to keep a candidate.
    min_trust : float
        Minimum trust score to keep a candidate.

    Returns
    -------
    ScoutReport
        Aggregated, sorted, filtered results.
    """
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    all_candidates: list[SearchResult] = []
    seen_repos: set[str] = set()
    filtered = 0

    for strategy in strategies:
        vocab = STRATEGIES.get(strategy)
        if not vocab:
            logger.warning("Unknown strategy %r — skipping", strategy)
            continue

        logger.info("Running strategy: %s (%d terms)", strategy, len(vocab))

        for term in vocab:
            query = f"{term} language:python"
            try:
                repos = search_github(query, sort="stars", max_results=30)
            except Exception as exc:
                logger.error("Search failed for term %r: %s", term, exc)
                continue

            for repo in repos:
                full_name = repo.get("full_name", "")
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)

                result = score_candidate(repo, strategy)

                if result.relevance_score < min_relevance or result.trust_score < min_trust:
                    filtered += 1
                    continue

                all_candidates.append(result)

    # Sort by relevance descending
    all_candidates.sort(key=lambda c: c.relevance_score, reverse=True)

    return ScoutReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        candidates=all_candidates,
        filtered_count=filtered,
    )


# ---------------------------------------------------------------------------
# CLI entry point (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    strats = sys.argv[1:] or None
    report = run_scout(strategies=strats)

    print(report.summary())
    print()
    for c in report.candidates[:20]:
        print(
            f"  {c.relevance_score:.2f} rel  {c.trust_score:.2f} trust  "
            f"{'*' * c.stars if c.stars < 6 else f'{c.stars}*'}  "
            f"{c.repo_full_name}  [{c.target_zone}]"
        )
        if c.query_terms_matched:
            print(f"    matched: {', '.join(c.query_terms_matched)}")
