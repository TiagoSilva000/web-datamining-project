"""Lab 4: entity linking + predicate alignment against Wikidata.

Outputs:
- entity_links.csv        mapping table with best candidate + heuristic confidence
- predicate_alignment.csv property suggestions
- alignment.ttl          owl:sameAs / owl:equivalentProperty for high-confidence matches

Notes:
- The Wikidata wbsearchentities API is used for entity lookup.
- The Wikidata Query Service is used to suggest matching properties.
- The confidence score is a transparent heuristic computed by this script.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef

DEFAULT_BASE_URI = "http://example.org/webdatamining/"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "WebMiningSemanticsProject/1.0 (student project)"
TYPE_HINTS = {
    "PERSON": ["actor", "director", "writer", "producer", "television", "film", "person", "screenwriter"],
    "ORG": ["organization", "company", "studio", "network", "award", "academy"],
    "GPE": ["city", "country", "state", "location", "place"],
    "WORK_OF_ART": ["film", "television", "series", "movie", "show", "work"],
    "EVENT": ["award", "ceremony", "event", "festival"],
    "DATE": ["year", "date", "time"],
}
PROPERTY_STOPWORDS = {"be", "have", "do", "get", "make", "go", "use", "say", "tell", "see", "in", "on", "at", "of"}


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "unnamed"


def to_camel_case(text: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", text) if p]
    if not parts:
        return "relatedTo"
    head = parts[0].lower()
    tail = "".join(p.capitalize() for p in parts[1:])
    return head + tail


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def token_set(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", normalize(text)) if t}


def build_private_entity_uri(base_uri: str, entity_text: str) -> str:
    if not base_uri.endswith("/"):
        base_uri += "/"
    return f"{base_uri}entity/{slugify(entity_text)}"


def build_private_property_uri(base_uri: str, predicate: str) -> str:
    if not base_uri.endswith("/"):
        base_uri += "/"
    return f"{base_uri}property/{to_camel_case(predicate)}"


def load_entities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"entity_normalized", "entity_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Entities CSV is missing required columns: {sorted(missing)}")
    return df


def load_relations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"predicate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Relations CSV is missing required columns: {sorted(missing)}")
    return df


def wikidata_search(query: str, *, limit: int, user_agent: str, entity_type: Optional[str] = None) -> list[dict]:
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": limit,
    }
    if entity_type:
        params["type"] = entity_type
    response = requests.get(
        WIKIDATA_API,
        params=params,
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("search", [])


def score_entity_candidate(entity_text: str, entity_type: str, candidate: dict, second_best_label: str | None = None) -> tuple[float, str]:
    entity_norm = normalize(entity_text)
    label = candidate.get("label", "") or ""
    label_norm = normalize(label)
    desc = normalize(candidate.get("description", "") or "")
    match_text = normalize((candidate.get("match") or {}).get("text", "") or "")
    overlap = len(token_set(entity_norm) & token_set(label_norm))
    denom = max(len(token_set(entity_norm)), 1)

    score = 0.05
    reasons: list[str] = []

    if label_norm == entity_norm:
        score += 0.60
        reasons.append("exact_label")
    elif entity_norm in label_norm or label_norm in entity_norm:
        score += 0.40
        reasons.append("partial_label")
    elif overlap:
        score += 0.25 * (overlap / denom)
        reasons.append("token_overlap")

    if match_text and match_text == entity_norm:
        score += 0.10
        reasons.append("api_match_text")

    for hint in TYPE_HINTS.get(entity_type, []):
        if hint in desc:
            score += 0.15
            reasons.append(f"type_hint:{hint}")
            break

    if second_best_label and normalize(second_best_label) == label_norm:
        score -= 0.05

    return min(max(score, 0.0), 0.99), ",".join(reasons) if reasons else "weak_match"


def align_entities(
    entities_df: pd.DataFrame,
    *,
    base_uri: str,
    min_confidence: float,
    user_agent: str,
    limit: int,
    delay: float,
) -> pd.DataFrame:
    unique_entities = (
        entities_df[["entity_normalized", "entity_type"]]
        .drop_duplicates()
        .sort_values(["entity_type", "entity_normalized"])
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    for _, row in unique_entities.iterrows():
        entity_text = str(row["entity_normalized"]).strip()
        entity_type = str(row["entity_type"]).strip()

        if not entity_text or entity_type == "DATE":
            rows.append(
                {
                    "private_entity": entity_text,
                    "private_uri": build_private_entity_uri(base_uri, entity_text),
                    "entity_type": entity_type,
                    "wikidata_id": "",
                    "wikidata_uri": "",
                    "wikidata_label": "",
                    "wikidata_description": "",
                    "confidence": 0.0,
                    "matched": False,
                    "match_reason": "skipped_date_or_empty",
                }
            )
            continue

        candidates = wikidata_search(entity_text, limit=limit, user_agent=user_agent)
        best_candidate = None
        best_score = -1.0
        best_reason = ""
        second_label = candidates[1].get("label") if len(candidates) > 1 else None

        for candidate in candidates:
            score, reason = score_entity_candidate(entity_text, entity_type, candidate, second_best_label=second_label)
            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_reason = reason

        matched = bool(best_candidate and best_score >= min_confidence)
        rows.append(
            {
                "private_entity": entity_text,
                "private_uri": build_private_entity_uri(base_uri, entity_text),
                "entity_type": entity_type,
                "wikidata_id": (best_candidate or {}).get("id", "") if matched else "",
                "wikidata_uri": (best_candidate or {}).get("concepturi", "") if matched else "",
                "wikidata_label": (best_candidate or {}).get("label", "") if matched else "",
                "wikidata_description": (best_candidate or {}).get("description", "") if matched else "",
                "confidence": round(best_score if best_score > 0 else 0.0, 4),
                "matched": matched,
                "match_reason": best_reason if matched else (best_reason or "below_threshold"),
            }
        )
        time.sleep(delay)

    return pd.DataFrame(rows)


def suggest_property_alignment(
    predicate: str,
    *,
    user_agent: str,
    limit: int,
) -> tuple[Optional[dict], float, str]:
    meaningful_tokens = [t for t in re.split(r"[^a-z0-9]+", predicate.lower()) if t and t not in PROPERTY_STOPWORDS]
    if not meaningful_tokens:
        return None, 0.0, "no_meaningful_tokens"

    search_term = max(meaningful_tokens, key=len)
    query = f"""
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>

SELECT ?property ?propertyLabel ?propertyDescription WHERE {{
  ?property a wikibase:Property ;
            rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
  OPTIONAL {{
    ?property schema:description ?propertyDescription .
    FILTER(LANG(?propertyDescription) = "en")
  }}
  FILTER(CONTAINS(LCASE(?propertyLabel), "{search_term.lower()}"))
}}
LIMIT {int(limit)}
"""
    response = requests.get(
        WDQS_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json", "User-Agent": user_agent},
        timeout=60,
    )
    response.raise_for_status()
    bindings = response.json().get("results", {}).get("bindings", [])

    best = None
    best_score = -1.0
    best_reason = ""

    pred_norm = normalize(predicate.replace("_", " "))
    pred_tokens = token_set(pred_norm)

    for item in bindings:
        label = item.get("propertyLabel", {}).get("value", "")
        label_norm = normalize(label)
        label_tokens = token_set(label_norm)
        overlap = len(pred_tokens & label_tokens)

        score = 0.05
        reasons = []
        if label_norm == pred_norm:
            score += 0.70
            reasons.append("exact_label")
        elif overlap:
            score += 0.35 * (overlap / max(len(pred_tokens), 1))
            reasons.append("token_overlap")
        if search_term in label_norm:
            score += 0.15
            reasons.append("search_token")
        desc = normalize(item.get("propertyDescription", {}).get("value", ""))
        if desc and any(tok in desc for tok in pred_tokens):
            score += 0.10
            reasons.append("description_overlap")

        if score > best_score:
            best = item
            best_score = score
            best_reason = ",".join(reasons) if reasons else "weak_match"

    return best, min(max(best_score, 0.0), 0.99), best_reason or "weak_match"


def align_predicates(
    relations_df: pd.DataFrame,
    *,
    base_uri: str,
    min_confidence: float,
    user_agent: str,
    limit: int,
    delay: float,
) -> pd.DataFrame:
    predicates = (
        relations_df["predicate"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    rows: list[dict] = []
    for predicate in predicates:
        best, confidence, reason = suggest_property_alignment(predicate, user_agent=user_agent, limit=limit)
        matched = bool(best and confidence >= min_confidence)
        rows.append(
            {
                "private_predicate": predicate,
                "private_uri": build_private_property_uri(base_uri, predicate),
                "wikidata_property": best.get("property", {}).get("value", "") if best and matched else "",
                "property_label": best.get("propertyLabel", {}).get("value", "") if best and matched else "",
                "property_description": best.get("propertyDescription", {}).get("value", "") if best and matched else "",
                "confidence": round(confidence, 4),
                "matched": matched,
                "match_reason": reason if matched else (reason or "below_threshold"),
            }
        )
        time.sleep(delay)

    return pd.DataFrame(rows)


def build_alignment_graph(
    entity_links_df: pd.DataFrame,
    predicate_links_df: pd.DataFrame,
) -> Graph:
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    for _, row in entity_links_df.iterrows():
        if not bool(row.get("matched")) or not str(row.get("wikidata_uri", "")).strip():
            continue
        g.add((URIRef(str(row["private_uri"])), OWL.sameAs, URIRef(str(row["wikidata_uri"]))))

    for _, row in predicate_links_df.iterrows():
        if not bool(row.get("matched")) or not str(row.get("wikidata_property", "")).strip():
            continue
        g.add((URIRef(str(row["private_uri"])), OWL.equivalentProperty, URIRef(str(row["wikidata_property"]))))

    return g


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align Lab 4 entities and predicates with Wikidata")
    parser.add_argument("--entities", type=Path, required=True, help="data/extracted_knowledge.csv")
    parser.add_argument("--relations", type=Path, required=True, help="data/relation_candidates.csv")
    parser.add_argument("--entity-links-out", type=Path, required=True, help="CSV mapping table for entities")
    parser.add_argument("--predicate-links-out", type=Path, required=True, help="CSV mapping table for predicates")
    parser.add_argument("--alignment-out", type=Path, required=True, help="Turtle file with owl:sameAs / owl:equivalentProperty")
    parser.add_argument("--base-uri", type=str, default=DEFAULT_BASE_URI, help="Base URI used in the private KB")
    parser.add_argument("--min-entity-confidence", type=float, default=0.72)
    parser.add_argument("--min-property-confidence", type=float, default=0.75)
    parser.add_argument("--search-limit", type=int, default=5, help="How many entity search candidates to inspect")
    parser.add_argument("--property-limit", type=int, default=20, help="How many property candidates to inspect")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between remote requests")
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    entities_df = load_entities(args.entities)
    relations_df = load_relations(args.relations)

    entity_links_df = align_entities(
        entities_df,
        base_uri=args.base_uri,
        min_confidence=args.min_entity_confidence,
        user_agent=args.user_agent,
        limit=args.search_limit,
        delay=args.delay,
    )
    predicate_links_df = align_predicates(
        relations_df,
        base_uri=args.base_uri,
        min_confidence=args.min_property_confidence,
        user_agent=args.user_agent,
        limit=args.property_limit,
        delay=args.delay,
    )

    alignment_graph = build_alignment_graph(entity_links_df, predicate_links_df)

    args.entity_links_out.parent.mkdir(parents=True, exist_ok=True)
    args.predicate_links_out.parent.mkdir(parents=True, exist_ok=True)
    args.alignment_out.parent.mkdir(parents=True, exist_ok=True)

    entity_links_df.to_csv(args.entity_links_out, index=False, quoting=csv.QUOTE_MINIMAL)
    predicate_links_df.to_csv(args.predicate_links_out, index=False, quoting=csv.QUOTE_MINIMAL)
    alignment_graph.serialize(destination=str(args.alignment_out), format="turtle")

    print(f"Saved entity links to {args.entity_links_out}")
    print(f"Saved predicate links to {args.predicate_links_out}")
    print(f"Saved alignment graph to {args.alignment_out}")
    print(f"Entity matches kept: {int(entity_links_df['matched'].sum())}/{len(entity_links_df)}")
    print(f"Predicate matches kept: {int(predicate_links_df['matched'].sum())}/{len(predicate_links_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
