"""Lab 4: expand the private KB with Wikidata one-hop / multi-hop triples.

The script reads the initial private KB + alignment results, then grows the graph
from confidently linked Wikidata seed entities using the Wikidata Query Service.

Only triples with entity IRIs on both sides are added, which makes the result
much cleaner for the later KGE lab.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import pandas as pd
import requests
from rdflib import Graph, URIRef

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "WebMiningSemanticsProject/1.0 (student project)"


def qid_from_uri(uri: str) -> str | None:
    match = re.search(r"/(Q\d+)$", uri or "")
    return match.group(1) if match else None


def query_one_hop_batch(qids: list[str], *, per_entity_limit: int, user_agent: str) -> list[tuple[str, str, str]]:
    if not qids:
        return []
    values = " ".join(f"wd:{qid}" for qid in qids)
    limit = max(len(qids) * per_entity_limit, 1)
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?s ?p ?o WHERE {{
  VALUES ?s {{ {values} }}
  ?s ?p ?o .
  FILTER(STRSTARTS(STR(?p), STR(wdt:)))
  FILTER(isIRI(?o))
  FILTER(STRSTARTS(STR(?o), STR(wd:)))
}}
LIMIT {limit}
"""
    response = requests.get(
        WDQS_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json", "User-Agent": user_agent},
        timeout=90,
    )
    response.raise_for_status()
    bindings = response.json().get("results", {}).get("bindings", [])
    triples: list[tuple[str, str, str]] = []
    for item in bindings:
        s = item.get("s", {}).get("value")
        p = item.get("p", {}).get("value")
        o = item.get("o", {}).get("value")
        if s and p and o:
            triples.append((s, p, o))
    return triples


def load_seed_qids(entity_links_csv: Path, min_confidence: float) -> list[str]:
    df = pd.read_csv(entity_links_csv)
    required = {"wikidata_uri", "confidence", "matched"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Entity links CSV is missing required columns: {sorted(missing)}")

    qids: list[str] = []
    for _, row in df.iterrows():
        if not bool(row.get("matched")):
            continue
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        qid = qid_from_uri(str(row.get("wikidata_uri", "")))
        if qid:
            qids.append(qid)

    # preserve order, remove duplicates
    return list(dict.fromkeys(qids))


def expand_graph(
    graph: Graph,
    *,
    seed_qids: list[str],
    target_triples: int,
    max_depth: int,
    per_entity_limit: int,
    batch_size: int,
    delay: float,
    user_agent: str,
) -> dict:
    frontier = deque(seed_qids)
    seen_qids = set(seed_qids)
    expanded_qids: set[str] = set()
    wikidata_triples_added = 0
    depth = 0

    while frontier and depth < max_depth and len(graph) < target_triples:
        current_level: list[str] = []
        while frontier and len(current_level) < batch_size:
            qid = frontier.popleft()
            if qid in expanded_qids:
                continue
            current_level.append(qid)

        if not current_level:
            depth += 1
            continue

        triples = query_one_hop_batch(current_level, per_entity_limit=per_entity_limit, user_agent=user_agent)
        next_qids: list[str] = []

        for s, p, o in triples:
            triple = (URIRef(s), URIRef(p), URIRef(o))
            if triple not in graph:
                graph.add(triple)
                wikidata_triples_added += 1

            oqid = qid_from_uri(o)
            if oqid and oqid not in seen_qids:
                seen_qids.add(oqid)
                next_qids.append(oqid)

            if len(graph) >= target_triples:
                break

        expanded_qids.update(current_level)
        for qid in next_qids:
            frontier.append(qid)

        time.sleep(delay)

        # Once we have expanded at least all current seeds in the queue once, count that as one hop.
        if not frontier or all(q in expanded_qids for q in list(frontier)[: min(len(frontier), batch_size)]):
            depth += 1

    return {
        "seed_entity_count": len(seed_qids),
        "expanded_entity_count": len(expanded_qids),
        "wikidata_entities_seen": len(seen_qids),
        "wikidata_triples_added": wikidata_triples_added,
        "final_total_triples": len(graph),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand the Lab 4 KB with Wikidata one-hop / multi-hop triples")
    parser.add_argument("--initial-kb", type=Path, required=True, help="kg_artifacts/initial_kb.ttl")
    parser.add_argument("--alignment", type=Path, required=True, help="kg_artifacts/alignment.ttl")
    parser.add_argument("--entity-links", type=Path, required=True, help="kg_artifacts/entity_links.csv")
    parser.add_argument("--output", type=Path, required=True, help="kg_artifacts/expanded.nt")
    parser.add_argument("--stats", type=Path, required=True, help="kg_artifacts/kb_stats_expanded.json")
    parser.add_argument("--target-triples", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--per-entity-limit", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--min-entity-confidence", type=float, default=0.72)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    graph = Graph()
    graph.parse(args.initial_kb, format="turtle")
    graph.parse(args.alignment, format="turtle")

    seed_qids = load_seed_qids(args.entity_links, min_confidence=args.min_entity_confidence)
    if not seed_qids:
        raise SystemExit("No confident aligned Wikidata entities found. Run align_wikidata.py first or lower the threshold.")

    stats = expand_graph(
        graph,
        seed_qids=seed_qids,
        target_triples=args.target_triples,
        max_depth=args.max_depth,
        per_entity_limit=args.per_entity_limit,
        batch_size=args.batch_size,
        delay=args.delay,
        user_agent=args.user_agent,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    graph.serialize(destination=str(args.output), format="nt")
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved expanded graph to {args.output}")
    print(f"Saved stats to {args.stats}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
