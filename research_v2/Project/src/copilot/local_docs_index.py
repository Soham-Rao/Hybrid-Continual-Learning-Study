"""Local document indexing across v1 and v2 workspaces for copilot retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from src.utils.paths import WORKSPACE_ROOT


TEXT_EXTENSIONS = {".md", ".txt"}
MAX_DOC_CHARS = 20000


@dataclass(frozen=True)
class LocalDocument:
    path: Path
    workspace: str
    label: str
    title: str
    text: str


def _candidate_roots() -> list[tuple[str, Path]]:
    return [
        ("research_v2", WORKSPACE_ROOT),
        ("v1_deadline_prototype", WORKSPACE_ROOT.parent / "v1_deadline_prototype"),
    ]


def _should_include(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return False
    lowered = str(path).lower()
    skip_parts = ["__pycache__", ".pytest_cache", "\\results\\runs\\", "\\results\\figures\\", "\\data_local\\"]
    return not any(part in lowered for part in skip_parts)


def _classify_label(path: Path, workspace: str) -> str:
    lowered = str(path).lower()
    if any(token in lowered for token in ["survey", "literature", "catastrophic", "cl methods", "hybrid", "machine learning"]):
        return "literature_note"
    if f"{workspace.lower()}\\docs\\" in lowered or "\\docs\\" in lowered or "\\my_docs\\" in lowered:
        return "design_note"
    return "design_note"


@lru_cache(maxsize=1)
def build_local_docs_index() -> tuple[LocalDocument, ...]:
    docs: list[LocalDocument] = []
    for workspace, root in _candidate_roots():
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _should_include(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not text:
                continue
            docs.append(
                LocalDocument(
                    path=path,
                    workspace=workspace,
                    label=_classify_label(path, workspace),
                    title=path.stem.replace("_", " "),
                    text=text[:MAX_DOC_CHARS],
                )
            )
    return tuple(sorted(docs, key=lambda item: str(item.path)))


def search_local_docs(query: str, limit: int = 5, *, labels: Iterable[str] | None = None) -> list[LocalDocument]:
    terms = [part.lower() for part in query.split() if part.strip()]
    label_filter = set(labels or [])
    ranked: list[tuple[int, LocalDocument]] = []
    for doc in build_local_docs_index():
        if label_filter and doc.label not in label_filter:
            continue
        haystack = f"{doc.title}\n{doc.text[:4000]}".lower()
        score = sum(haystack.count(term) for term in terms) if terms else 1
        if score <= 0:
            continue
        ranked.append((score, doc))
    ranked.sort(key=lambda item: (-item[0], len(str(item[1].path))))
    return [doc for _, doc in ranked[:limit]]
