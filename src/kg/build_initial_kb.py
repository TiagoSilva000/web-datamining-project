"""Lab 4: build the initial private knowledge base in RDF.

This script turns Lab 1 outputs into:
- an initial RDF graph (Turtle)
- an ontology file (Turtle)
- a small statistics report (JSON)

It expects the CSV schema produced by the Lab 1 extraction script:
  extracted_knowledge.csv
  relation_candidates.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

DEFAULT_BASE_URI = "http://example.org/webdatamining/"
TYPE_TO_CLASS = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "WORK_OF_ART": "Work",
    "EVENT": "Event",
    "DATE": "DateEntity",
}
CLASS_HIERARCHY = {
    "Person": "Agent",
    "Organization": "Agent",
    "Place": None,
    "Work": None,
    "Event": None,
    "DateEntity": None,
    "Agent": None,
}
STOPWORDS = {"be", "have", "do", "get", "make", "go"}


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
    candidate = head + tail
    if candidate in STOPWORDS:
        return f"{candidate}Relation"
    return candidate


def guess_class(entity_type: Optional[str]) -> str:
    return TYPE_TO_CLASS.get((entity_type or "").strip(), "Thing")


def build_namespaces(base_uri: str) -> dict[str, Namespace]:
    if not base_uri.endswith("/"):
        base_uri += "/"
    return {
        "EX": Namespace(base_uri),
        "ENT": Namespace(base_uri + "entity/"),
        "PROP": Namespace(base_uri + "property/"),
        "ONT": Namespace(base_uri + "ontology/"),
    }


def ensure_path_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def entity_uri(ns: dict[str, Namespace], text: str) -> URIRef:
    return ns["ENT"][slugify(text)]


def class_uri(ns: dict[str, Namespace], name: str) -> URIRef:
    return ns["ONT"][name]


def prop_uri(ns: dict[str, Namespace], predicate_text: str) -> URIRef:
    return ns["PROP"][to_camel_case(predicate_text)]


def add_ontology_shell(ontology_graph: Graph, ns: dict[str, Namespace]) -> None:
    ontology_graph.bind("ex", ns["EX"])
    ontology_graph.bind("ent", ns["ENT"])
    ontology_graph.bind("prop", ns["PROP"])
    ontology_graph.bind("ont", ns["ONT"])
    ontology_graph.bind("rdf", RDF)
    ontology_graph.bind("rdfs", RDFS)
    ontology_graph.bind("owl", OWL)

    for cls_name, parent in CLASS_HIERARCHY.items():
        cls = class_uri(ns, cls_name)
        ontology_graph.add((cls, RDF.type, OWL.Class))
        ontology_graph.add((cls, RDFS.label, Literal(cls_name)))
        if parent:
            ontology_graph.add((cls, RDFS.subClassOf, class_uri(ns, parent)))


def load_entities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "entity_text",
        "entity_normalized",
        "entity_type",
        "source_url",
        "mention_count",
        "example_sentence",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Entities CSV is missing required columns: {sorted(missing)}")
    return df


def load_relations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "subject",
        "subject_type",
        "predicate",
        "object",
        "object_type",
        "source_url",
        "sentence",
        "confidence",
        "method",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Relations CSV is missing required columns: {sorted(missing)}")
    return df


def build_initial_graph(
    entities_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    base_uri: str,
    min_relation_confidence: float,
) -> tuple[Graph, Graph, dict]:
    ns = build_namespaces(base_uri)

    g = Graph()
    g.bind("ex", ns["EX"])
    g.bind("ent", ns["ENT"])
    g.bind("prop", ns["PROP"])
    g.bind("ont", ns["ONT"])
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    ontology = Graph()
    add_ontology_shell(ontology, ns)

    # Aggregate entity mentions across sources.
    entity_meta: dict[str, dict] = {}
    for _, row in entities_df.iterrows():
        entity_text = str(row["entity_normalized"]).strip() or str(row["entity_text"]).strip()
        if not entity_text:
            continue
        key = entity_text
        meta = entity_meta.setdefault(
            key,
            {
                "labels": Counter(),
                "types": Counter(),
                "mentions": 0,
                "sources": set(),
            },
        )
        meta["labels"][str(row["entity_text"]).strip() or entity_text] += 1
        meta["types"][str(row["entity_type"]).strip()] += 1
        meta["mentions"] += int(row.get("mention_count", 1) or 1)
        meta["sources"].add(str(row["source_url"]).strip())

    # Make sure entities found only in relation candidates also exist.
    for _, row in relations_df.iterrows():
        for side, side_type in [("subject", "subject_type"), ("object", "object_type")]:
            text = str(row[side]).strip()
            if not text:
                continue
            meta = entity_meta.setdefault(
                text,
                {"labels": Counter(), "types": Counter(), "mentions": 0, "sources": set()},
            )
            meta["labels"][text] += 1
            meta["types"][str(row[side_type]).strip()] += 1
            meta["sources"].add(str(row["source_url"]).strip())

    # Add entity nodes.
    for entity_text, meta in sorted(entity_meta.items()):
        ent = entity_uri(ns, entity_text)
        g.add((ent, RDF.type, OWL.NamedIndividual))
        g.add((ent, RDFS.label, Literal(meta["labels"].most_common(1)[0][0])))
        g.add((ent, ns["EX"]["mentionCount"], Literal(meta["mentions"], datatype=XSD.integer)))

        observed_types = [t for t, _ in meta["types"].most_common() if t]
        if not observed_types:
            observed_types = ["Thing"]
        for observed in observed_types:
            cls_name = guess_class(observed)
            g.add((ent, RDF.type, class_uri(ns, cls_name)))

    # Aggregate observed domain/range per predicate from relation candidates.
    pred_domains: dict[str, Counter] = defaultdict(Counter)
    pred_ranges: dict[str, Counter] = defaultdict(Counter)
    kept_relations = 0
    skipped_relations = 0

    seen_relation_triples: set[tuple[str, str, str]] = set()
    for _, row in relations_df.iterrows():
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if confidence < min_relation_confidence:
            skipped_relations += 1
            continue

        subj_text = str(row["subject"]).strip()
        obj_text = str(row["object"]).strip()
        pred_text = str(row["predicate"]).strip()
        if not subj_text or not obj_text or not pred_text:
            skipped_relations += 1
            continue

        triple_key = (subj_text, pred_text, obj_text)
        if triple_key in seen_relation_triples:
            continue
        seen_relation_triples.add(triple_key)

        subj = entity_uri(ns, subj_text)
        obj = entity_uri(ns, obj_text)
        pred = prop_uri(ns, pred_text)

        g.add((subj, pred, obj))
        kept_relations += 1

        pred_domains[pred_text][guess_class(str(row["subject_type"]).strip())] += 1
        pred_ranges[pred_text][guess_class(str(row["object_type"]).strip())] += 1

    # Create ontology entries for relation predicates.
    for pred_text in sorted(pred_domains.keys() | pred_ranges.keys()):
        prop = prop_uri(ns, pred_text)
        ontology.add((prop, RDF.type, OWL.ObjectProperty))
        ontology.add((prop, RDFS.label, Literal(pred_text)))

        if pred_domains[pred_text]:
            domain_cls = pred_domains[pred_text].most_common(1)[0][0]
            if domain_cls != "Thing":
                ontology.add((prop, RDFS.domain, class_uri(ns, domain_cls)))
        if pred_ranges[pred_text]:
            range_cls = pred_ranges[pred_text].most_common(1)[0][0]
            if range_cls != "Thing":
                ontology.add((prop, RDFS.range, class_uri(ns, range_cls)))

    stats = {
        "total_triples": len(g),
        "entity_count": len(entity_meta),
        "relation_count": len({str(prop_uri(ns, p)) for p in pred_domains.keys()}),
        "kept_relation_assertions": kept_relations,
        "filtered_low_confidence_relations": skipped_relations,
        "base_uri": base_uri,
    }
    return g, ontology, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the initial private RDF KB for Lab 4")
    parser.add_argument("--entities", type=Path, required=True, help="data/extracted_knowledge.csv")
    parser.add_argument("--relations", type=Path, required=True, help="data/relation_candidates.csv")
    parser.add_argument("--initial-kb", type=Path, required=True, help="Output Turtle file for the initial KB")
    parser.add_argument("--ontology", type=Path, required=True, help="Output Turtle file for the ontology")
    parser.add_argument("--stats", type=Path, required=True, help="Output JSON stats report")
    parser.add_argument("--base-uri", type=str, default=DEFAULT_BASE_URI, help="Base URI for private resources")
    parser.add_argument(
        "--min-relation-confidence",
        type=float,
        default=0.50,
        help="Keep only relation candidates at or above this confidence",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    entities_df = load_entities(args.entities)
    relations_df = load_relations(args.relations)

    graph, ontology, stats = build_initial_graph(
        entities_df=entities_df,
        relations_df=relations_df,
        base_uri=args.base_uri,
        min_relation_confidence=args.min_relation_confidence,
    )

    ensure_path_parent(args.initial_kb)
    ensure_path_parent(args.ontology)
    ensure_path_parent(args.stats)

    graph.serialize(destination=str(args.initial_kb), format="turtle")
    ontology.serialize(destination=str(args.ontology), format="turtle")
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved initial KB to {args.initial_kb}")
    print(f"Saved ontology to {args.ontology}")
    print(f"Saved stats to {args.stats}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
