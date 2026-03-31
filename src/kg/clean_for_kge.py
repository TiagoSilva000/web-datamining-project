from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Set

from rdflib import Graph, URIRef, Literal

BLOCKED_PREDICATE_SUBSTRINGS = [
    "rdf-schema#label",
    "schema.org/description",
    "schema.org/name",
    "schema.org/dateModified",
    "schema.org/version",
    "wikiba.se/ontology#sitelinks",
    "wikiba.se/ontology#identifiers",
    "wikiba.se/ontology#statements",
    "wikiba.se/ontology#timestamp",
    "wikiba.se/ontology#badge",
    "specialEntityData",
    "prov#",
    "skos/core#altLabel",
]


def is_blocked_predicate(p: URIRef) -> bool:
    ps = str(p)
    return any(token in ps for token in BLOCKED_PREDICATE_SUBSTRINGS)


def load_graph(path: Path) -> Graph:
    g = Graph()
    fmt = "nt" if path.suffix.lower() == ".nt" else "turtle"
    g.parse(path, format=fmt)
    return g


def build_filtered_graph(g: Graph, keep_types: bool = True) -> tuple[Graph, dict]:
    out = Graph()
    dropped = Counter()

    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

    for s, p, o in g:
        if not isinstance(s, URIRef):
            dropped["non_uri_subject"] += 1
            continue
        if not isinstance(p, URIRef):
            dropped["non_uri_predicate"] += 1
            continue
        if isinstance(o, Literal):
            dropped["literal_object"] += 1
            continue
        if not isinstance(o, URIRef):
            dropped["non_uri_object"] += 1
            continue
        if is_blocked_predicate(p):
            dropped["blocked_predicate"] += 1
            continue
        if not keep_types and p == rdf_type:
            dropped["rdf_type_removed"] += 1
            continue
        out.add((s, p, o))

    return out, dict(dropped)


def largest_component_nodes(g: Graph) -> Set[URIRef]:
    adj = defaultdict(set)
    nodes = set()
    for s, _, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            adj[s].add(o)
            adj[o].add(s)
            nodes.add(s)
            nodes.add(o)

    seen = set()
    best = set()

    for start in nodes:
        if start in seen:
            continue
        comp = set()
        q = deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def keep_only_component(g: Graph, allowed_nodes: Set[URIRef]) -> Graph:
    out = Graph()
    for s, p, o in g:
        if s in allowed_nodes and o in allowed_nodes:
            out.add((s, p, o))
    return out


def relation_count(g: Graph) -> int:
    return len({str(p) for _, p, _ in g})


def entity_count(g: Graph) -> int:
    ents = set()
    for s, _, o in g:
        ents.add(str(s))
        ents.add(str(o))
    return len(ents)


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean expanded KG for KGE")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--drop-rdf-type", action="store_true")
    ap.add_argument("--largest-component", action="store_true")
    args = ap.parse_args()

    g = load_graph(args.input)
    filtered, dropped = build_filtered_graph(g, keep_types=not args.drop_rdf_type)

    component_nodes = None
    if args.largest_component:
        component_nodes = largest_component_nodes(filtered)
        filtered = keep_only_component(filtered, component_nodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(filtered.serialize(format="nt"), encoding="utf-8")

    stats = {
        "input_triples": len(g),
        "output_triples": len(filtered),
        "entity_count": entity_count(filtered),
        "relation_count": relation_count(filtered),
        "dropped": dropped,
        "largest_component_nodes": len(component_nodes) if component_nodes is not None else None,
    }
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved cleaned KG to {args.output}")
    print(f"Saved cleaning stats to {args.stats}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
