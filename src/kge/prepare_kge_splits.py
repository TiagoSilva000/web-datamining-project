"""Prepare leakage-resistant KGE train/valid/test splits from an N-Triples graph.

Expected input: an RDF graph (e.g. kg_artifacts/expanded_clean.nt) where triples are
mostly URI-URI triples. Literal-object triples are ignored.

Outputs:
- train.txt / valid.txt / test.txt (tab-separated head, relation, tail)
- a JSON stats file
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rdflib import Graph, URIRef

Triple = Tuple[str, str, str]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare train/valid/test KGE splits")
    parser.add_argument("--input", type=Path, required=True, help="Input RDF/NT graph")
    parser.add_argument("--train-out", type=Path, required=True, help="Output TSV for training triples")
    parser.add_argument("--valid-out", type=Path, required=True, help="Output TSV for validation triples")
    parser.add_argument("--test-out", type=Path, required=True, help="Output TSV for test triples")
    parser.add_argument("--stats-out", type=Path, required=True, help="Where to save split statistics JSON")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-triples",
        type=int,
        default=0,
        help="Optional cap on number of triples to use before splitting (0 means use all)",
    )
    return parser


def load_uri_triples(path: Path) -> List[Triple]:
    graph = Graph()
    # RDFLib can infer N-Triples from .nt
    graph.parse(str(path))
    triples: set[Triple] = set()
    for s, p, o in graph:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            triples.add((str(s), str(p), str(o)))
    if not triples:
        raise ValueError(f"No URI-URI triples found in {path}")
    return sorted(triples)


def write_tsv(path: Path, triples: Sequence[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for h, r, t in triples:
            handle.write(f"{h}\t{r}\t{t}\n")


def count_entities_relations(triples: Iterable[Triple]) -> tuple[Counter, Counter]:
    entity_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    for h, r, t in triples:
        entity_counter[h] += 1
        entity_counter[t] += 1
        relation_counter[r] += 1
    return entity_counter, relation_counter


def safe_split(
    triples: List[Triple],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[List[Triple], List[Triple], List[Triple]]:
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    triples = list(triples)
    random.Random(seed).shuffle(triples)

    total = len(triples)
    target_valid = int(total * valid_ratio)
    target_test = int(total * test_ratio)

    train = list(triples)
    valid: List[Triple] = []
    test: List[Triple] = []

    entity_counter, relation_counter = count_entities_relations(train)

    def can_remove(triple: Triple) -> bool:
        h, r, t = triple
        return entity_counter[h] > 1 and entity_counter[t] > 1 and relation_counter[r] > 1

    idx = 0
    # Fill validation first
    while len(valid) < target_valid and idx < len(train):
        triple = train[idx]
        if can_remove(triple):
            h, r, t = triple
            entity_counter[h] -= 1
            entity_counter[t] -= 1
            relation_counter[r] -= 1
            valid.append(triple)
            train.pop(idx)
        else:
            idx += 1

    idx = 0
    # Fill test second
    while len(test) < target_test and idx < len(train):
        triple = train[idx]
        if can_remove(triple):
            h, r, t = triple
            entity_counter[h] -= 1
            entity_counter[t] -= 1
            relation_counter[r] -= 1
            test.append(triple)
            train.pop(idx)
        else:
            idx += 1

    # Safety pass: move back any leakage-causing triples from valid/test to train.
    train_entities = {h for h, _, _ in train} | {t for _, _, t in train}
    train_relations = {r for _, r, _ in train}

    def filter_leakage(candidates: List[Triple]) -> List[Triple]:
        kept: List[Triple] = []
        for triple in candidates:
            h, r, t = triple
            if h in train_entities and t in train_entities and r in train_relations:
                kept.append(triple)
            else:
                train.append(triple)
                train_entities.update([h, t])
                train_relations.add(r)
        return kept

    valid = filter_leakage(valid)
    test = filter_leakage(test)

    return train, valid, test


def unique_counts(triples: Sequence[Triple]) -> dict[str, int]:
    entities = {h for h, _, _ in triples} | {t for _, _, t in triples}
    relations = {r for _, r, _ in triples}
    return {
        "triple_count": len(triples),
        "entity_count": len(entities),
        "relation_count": len(relations),
    }


def main() -> int:
    args = make_parser().parse_args()

    triples = load_uri_triples(args.input)
    original_count = len(triples)

    if args.max_triples and args.max_triples < len(triples):
        rng = random.Random(args.seed)
        triples = rng.sample(triples, k=args.max_triples)

    train, valid, test = safe_split(
        triples=triples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_tsv(args.train_out, train)
    write_tsv(args.valid_out, valid)
    write_tsv(args.test_out, test)

    stats = {
        "input_file": str(args.input),
        "original_uri_triple_count": original_count,
        "used_uri_triple_count": len(triples),
        "train": unique_counts(train),
        "valid": unique_counts(valid),
        "test": unique_counts(test),
        "seed": args.seed,
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved train split to {args.train_out}")
    print(f"Saved valid split to {args.valid_out}")
    print(f"Saved test split to {args.test_out}")
    print(f"Saved split stats to {args.stats_out}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
