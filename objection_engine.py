"""Generic objection-handling helper for the local RAG template."""
from __future__ import annotations

import re
from typing import Any


OBJECTION_TYPES = [
    "Price concern",
    "Value uncertainty",
    "Risk concern",
    "Competitor comparison",
    "No experience",
    "Budget or procurement barrier",
    "Implementation concern",
    "Evidence gap",
]


KEYWORD_RULES = [
    ("Price concern", ["price", "expensive", "cost", "budget", "pricing", "too high"]),
    ("Budget or procurement barrier", ["procurement", "purchase", "approval", "budget cycle"]),
    ("Risk concern", ["risk", "safety", "security", "privacy", "compliance", "concern"]),
    ("Competitor comparison", ["competitor", "alternative", "already use", "compare", "switch"]),
    ("Implementation concern", ["setup", "integration", "deployment", "migration", "training"]),
    ("Value uncertainty", ["value", "benefit", "roi", "effective", "worth"]),
    ("Evidence gap", ["evidence", "proof", "case study", "data", "reference"]),
    ("No experience", ["new", "unfamiliar", "never used", "first time"]),
]


OBJECTION_QUERY_EXPANSIONS = {
    "Price concern": [
        "pricing value total cost",
        "budget impact support resources",
        "long-term value operational efficiency",
    ],
    "Value uncertainty": [
        "value proposition measurable outcomes",
        "benefits use cases",
        "workflow efficiency examples",
    ],
    "Risk concern": [
        "security privacy risk controls",
        "limitations safeguards",
        "compliance data handling",
    ],
    "Competitor comparison": [
        "competitor comparison differentiation",
        "alternative workflow comparison",
        "switching considerations",
    ],
    "No experience": [
        "getting started guide",
        "training onboarding support",
        "first use workflow",
    ],
    "Budget or procurement barrier": [
        "procurement budget approval",
        "implementation support",
        "total cost value",
    ],
    "Implementation concern": [
        "setup deployment integration",
        "migration checklist",
        "training rollout",
    ],
    "Evidence gap": [
        "case study evidence examples",
        "reference data",
        "documented limitations",
    ],
}


EVIDENCE_KEYWORDS = [
    "pricing",
    "value",
    "risk",
    "security",
    "privacy",
    "workflow",
    "integration",
    "support",
    "training",
    "evidence",
    "example",
]


OBJECTION_SYSTEM_PROMPT = """You are a careful sales enablement coach helping draft a response to a customer concern.

Use only the supplied source material. Do not invent claims, data, case studies, or guarantees.

Use this fixed structure:
1. Concern identified
2. Response strategy
3. Suggested wording
4. Source basis
5. Cautions / claims to avoid

Requirements:
- Stay grounded in the supplied material.
- If direct support is missing, say so plainly.
- Cite filenames and page/slide/title details when available.
- Keep the tone professional, specific, and measured."""


def normalize_objection(text: str) -> str:
    """Normalize case and whitespace for simple keyword matching."""
    return re.sub(r"\s+", "", text or "").lower()


def classify_objection(objection: str) -> str:
    """
    Classify a customer objection into a generic template category.
    """
    normalized = normalize_objection(objection)
    if not normalized:
        return "Unclassified"

    for objection_type, keywords in KEYWORD_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            return objection_type
    return "Unclassified"


def expand_queries(objection: str, objection_type: str | None = None, max_queries: int = 8) -> list[str]:
    """Expand the original concern into multiple retrieval queries."""
    objection_type = objection_type or classify_objection(objection)
    queries = [objection.strip()] if objection.strip() else []
    queries.extend(OBJECTION_QUERY_EXPANSIONS.get(objection_type, []))

    seen = set()
    unique_queries = []
    for query in queries:
        normalized = normalize_objection(query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(query)
        if len(unique_queries) >= max_queries:
            break
    return unique_queries


def retrieve_evidence(
    store: Any,
    objection: str,
    objection_type: str | None = None,
    per_query_top_k: int = 5,
    final_top_k: int = 10,
) -> list[dict]:
    """Run multiple searches, merge, and deduplicate results."""
    queries = expand_queries(objection, objection_type)
    merged = {}

    for query in queries:
        for result in store.search(query, top_k=per_query_top_k):
            key = result.get("id") or (result.get("source"), result.get("page"), result.get("slide"), result.get("chunk_index"))
            if key not in merged:
                merged[key] = {**result, "matched_queries": [query]}
                continue
            merged[key]["score"] = max(merged[key].get("score", 0), result.get("score", 0))
            merged[key]["matched_queries"].append(query)

    return rerank_evidence(list(merged.values()), objection_type)[:final_top_k]


def rerank_evidence(results: list[dict], objection_type: str | None = None) -> list[dict]:
    """Rerank evidence with simple source and keyword boosts."""
    objection_filename = f"{objection_type}.md" if objection_type and objection_type != "Unclassified" else ""

    def score(result: dict) -> float:
        value = float(result.get("score", 0))
        source = result.get("source", "")
        source_folder = result.get("source_folder", "")
        text = result.get("text", "").lower()

        if source_folder == "sample_objections":
            value += 0.25
        if objection_filename and source == objection_filename:
            value += 0.2
        if "q&a" in source.lower() or "qa" in source.lower():
            value += 0.15
        value += min(sum(0.04 for keyword in EVIDENCE_KEYWORDS if keyword.lower() in text), 0.24)
        return value

    return sorted(results, key=score, reverse=True)


def format_evidence_context(results: list[dict], max_chars: int = 6000) -> str:
    """Format evidence chunks into cited context for the LLM."""
    blocks = []
    total = 0
    for index, result in enumerate(results, 1):
        citation = result.get("citation", result.get("source", "Unknown source"))
        text = result.get("text", "").strip()
        if not text:
            continue
        block = f"[{index}] {citation}\n{text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def generate_objection_response(objection: str, objection_type: str, evidence: list[dict]) -> dict:
    """Generate a fixed-structure objection response from evidence."""
    from llm_api import chat as llm_chat

    context = format_evidence_context(evidence)
    prompt = f"""Customer concern: {objection}
Identified objection type: {objection_type}

Available source material:
{context}
"""
    return llm_chat(prompt, OBJECTION_SYSTEM_PROMPT, temperature=0.4)
