"""Train and evaluate KGE models for Lab 5 using PyKEEN.

Features:
- uses pre-split train/valid/test files, or resamples them for size-sensitivity runs
- trains at least two models (default: TransE and ComplEx)
- saves MRR / Hits@1 / Hits@3 / Hits@10 for both/head/tail
- saves a summary CSV/JSON
- saves nearest-neighbor analysis for the best run
- saves a t-SNE plot for the best run
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdflib import Graph, URIRef
from sklearn.manifold import TSNE

try:
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pykeen and torch are required. Install them with: py -m pip install pykeen torch"
    ) from exc

Triple = Tuple[str, str, str]
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate KGE models with PyKEEN")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--graph-file", type=Path, default=None, help="Optional cleaned graph for entity type labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["TransE", "ComplEx"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[20000, 50000, 0], help="0 means full graph")
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-negs-per-pos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne-sample-size", type=int, default=1200)
    return parser


def load_tsv(path: Path) -> List[Triple]:
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Expected 3 tab-separated columns in {path}, got: {line}")
            triples.append(tuple(parts))  # type: ignore[arg-type]
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def count_entities_relations(triples: Iterable[Triple]) -> tuple[Counter, Counter]:
    entity_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    for h, r, t in triples:
        entity_counter[h] += 1
        entity_counter[t] += 1
        relation_counter[r] += 1
    return entity_counter, relation_counter


def safe_split(triples: List[Triple], seed: int) -> tuple[List[Triple], List[Triple], List[Triple]]:
    triples = list(dict.fromkeys(triples))
    random.Random(seed).shuffle(triples)

    total = len(triples)
    target_valid = int(total * 0.10)
    target_test = int(total * 0.10)

    train = list(triples)
    valid: List[Triple] = []
    test: List[Triple] = []

    entity_counter, relation_counter = count_entities_relations(train)

    def can_remove(triple: Triple) -> bool:
        h, r, t = triple
        return entity_counter[h] > 1 and entity_counter[t] > 1 and relation_counter[r] > 1

    idx = 0
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

    train_entities = {h for h, _, _ in train} | {t for _, _, t in train}
    train_relations = {r for _, r, _ in train}

    def fix_leakage(candidates: List[Triple]) -> List[Triple]:
        kept = []
        for triple in candidates:
            h, r, t = triple
            if h in train_entities and t in train_entities and r in train_relations:
                kept.append(triple)
            else:
                train.append(triple)
                train_entities.update([h, t])
                train_relations.add(r)
        return kept

    valid = fix_leakage(valid)
    test = fix_leakage(test)
    return train, valid, test


def to_factory(
    train: Sequence[Triple],
    valid: Sequence[Triple],
    test: Sequence[Triple],
) -> tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    train_arr = np.asarray(train, dtype=str)
    valid_arr = np.asarray(valid, dtype=str)
    test_arr = np.asarray(test, dtype=str)

    tf_train = TriplesFactory.from_labeled_triples(train_arr, create_inverse_triples=False)
    tf_valid = TriplesFactory.from_labeled_triples(
        valid_arr,
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
        create_inverse_triples=False,
    )
    tf_test = TriplesFactory.from_labeled_triples(
        test_arr,
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
        create_inverse_triples=False,
    )
    return tf_train, tf_valid, tf_test


def get_metric(metric_results, *names: str) -> float | None:
    for name in names:
        try:
            return float(metric_results.get_metric(name))
        except Exception:
            continue
    return None


def simplify_uri(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def extract_entity_embeddings(model) -> np.ndarray:
    reps = getattr(model, "entity_representations", None)
    if reps is None:
        raise RuntimeError("Model does not expose entity_representations")
    rep0 = reps[0]
    try:
        tensor = rep0()
    except TypeError:
        tensor = rep0(indices=None)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    return np.asarray(tensor)


def extract_relation_embeddings(model) -> np.ndarray:
    reps = getattr(model, "relation_representations", None)
    if reps is None:
        raise RuntimeError("Model does not expose relation_representations")
    rep0 = reps[0]
    try:
        tensor = rep0()
    except TypeError:
        tensor = rep0(indices=None)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    return np.asarray(tensor)


def save_embeddings(run_dir: Path, tf_train: TriplesFactory, model) -> None:
    entity_embeddings = extract_entity_embeddings(model)
    relation_embeddings = extract_relation_embeddings(model)

    np.save(run_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(run_dir / "relation_embeddings.npy", relation_embeddings)
    (run_dir / "entity_to_id.json").write_text(json.dumps(tf_train.entity_to_id, indent=2), encoding="utf-8")
    (run_dir / "relation_to_id.json").write_text(json.dumps(tf_train.relation_to_id, indent=2), encoding="utf-8")


def cosine_similarity_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    vector_norm = vector / np.clip(np.linalg.norm(vector), 1e-12, None)
    return matrix_norm @ vector_norm


def degree_ranked_entities(triples: Sequence[Triple]) -> list[str]:
    counter = Counter()
    for h, _, t in triples:
        counter[h] += 1
        counter[t] += 1
    return [entity for entity, _ in counter.most_common()]


def read_type_labels(graph_file: Path | None) -> dict[str, str]:
    if graph_file is None:
        return {}
    graph = Graph()
    graph.parse(str(graph_file))
    labels: dict[str, str] = {}
    for s, p, o in graph:
        if str(p) == RDF_TYPE and isinstance(s, URIRef) and isinstance(o, URIRef):
            labels.setdefault(str(s), simplify_uri(str(o)))
    return labels


def write_nearest_neighbors(
    run_dir: Path,
    tf_train: TriplesFactory,
    entity_embeddings: np.ndarray,
    reference_entities: Sequence[str],
    top_k: int = 10,
) -> None:
    id_to_entity = {idx: entity for entity, idx in tf_train.entity_to_id.items()}
    lines = []
    for entity in reference_entities[:10]:
        if entity not in tf_train.entity_to_id:
            continue
        idx = tf_train.entity_to_id[entity]
        sims = cosine_similarity_matrix(entity_embeddings, entity_embeddings[idx])
        order = np.argsort(-sims)
        neighbors = []
        for neighbor_idx in order:
            if neighbor_idx == idx:
                continue
            neighbors.append((id_to_entity[int(neighbor_idx)], float(sims[neighbor_idx])))
            if len(neighbors) >= top_k:
                break
        lines.append(f"Entity: {entity}")
        for nb_uri, score in neighbors:
            lines.append(f"  - {nb_uri} | cosine={score:.4f}")
        lines.append("")
    (run_dir / "nearest_neighbors.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_tsne(
    run_dir: Path,
    tf_train: TriplesFactory,
    entity_embeddings: np.ndarray,
    type_labels: dict[str, str],
    sample_size: int,
    seed: int,
) -> None:
    entities = list(tf_train.entity_to_id.keys())
    rng = random.Random(seed)
    candidates = [e for e in entities if e in type_labels]
    if len(candidates) < 50:
        candidates = entities
    if len(candidates) > sample_size:
        sampled_entities = rng.sample(candidates, sample_size)
    else:
        sampled_entities = candidates

    if len(sampled_entities) < 2:
        return

    indices = [tf_train.entity_to_id[e] for e in sampled_entities]
    matrix = entity_embeddings[indices]
    perplexity = min(30, max(5, len(sampled_entities) // 10))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    coords = tsne.fit_transform(matrix)

    labels = [type_labels.get(e, "Unknown") for e in sampled_entities]
    top_types = {label for label, _ in Counter(labels).most_common(8)}

    plt.figure(figsize=(12, 9))
    for label in sorted(top_types):
        xs = [coords[i, 0] for i, current in enumerate(labels) if current == label]
        ys = [coords[i, 1] for i, current in enumerate(labels) if current == label]
        plt.scatter(xs, ys, s=16, alpha=0.75, label=label)
    xs = [coords[i, 0] for i, current in enumerate(labels) if current not in top_types]
    ys = [coords[i, 1] for i, current in enumerate(labels) if current not in top_types]
    if xs:
        plt.scatter(xs, ys, s=12, alpha=0.4, label="Other")
    plt.title("t-SNE of entity embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "entity_tsne.png", dpi=180)
    plt.close()


def main() -> int:
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train = load_tsv(args.train)
    base_valid = load_tsv(args.valid)
    base_test = load_tsv(args.test)
    full_triples = list(dict.fromkeys(base_train + base_valid + base_test))
    full_count = len(full_triples)

    size_requests = []
    for size in args.sizes:
        resolved = full_count if size == 0 or size >= full_count else size
        size_requests.append(resolved)
    size_requests = list(dict.fromkeys(size_requests))

    results_rows = []
    type_labels = read_type_labels(args.graph_file)
    best_run = None

    for size in size_requests:
        subset = full_triples if size >= full_count else random.Random(args.seed + size).sample(full_triples, size)
        train, valid, test = safe_split(subset, seed=args.seed + size)
        tf_train, tf_valid, tf_test = to_factory(train, valid, test)
        size_dir = args.output_dir / f"size_{len(subset)}"
        size_dir.mkdir(parents=True, exist_ok=True)

        split_stats = {
            "train_triples": len(train),
            "valid_triples": len(valid),
            "test_triples": len(test),
            "entity_count": len(tf_train.entity_to_id),
            "relation_count": len(tf_train.relation_to_id),
        }
        (size_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")

        for model_name in args.models:
            run_dir = size_dir / model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            result = pipeline(
                training=tf_train,
                validation=tf_valid,
                testing=tf_test,
                model=model_name,
                model_kwargs={"embedding_dim": args.embedding_dim},
                optimizer="Adam",
                optimizer_kwargs={"lr": args.learning_rate},
                training_kwargs={"num_epochs": args.epochs, "batch_size": args.batch_size},
                training_loop="sLCWA",
                negative_sampler="basic",
                negative_sampler_kwargs={"num_negs_per_pos": args.num_negs_per_pos},
                evaluator="RankBasedEvaluator",
                evaluator_kwargs={"filtered": True},
                random_seed=args.seed,
                device="cpu",
            )
            result.save_to_directory(run_dir)

            metrics = result.metric_results
            row = {
                "size": len(subset),
                "model": model_name,
                "both_mrr": get_metric(metrics, "both.realistic.mean_reciprocal_rank", "mean_reciprocal_rank"),
                "head_mrr": get_metric(metrics, "head.realistic.mean_reciprocal_rank"),
                "tail_mrr": get_metric(metrics, "tail.realistic.mean_reciprocal_rank"),
                "both_hits@1": get_metric(metrics, "both.realistic.hits@1", "hits@1"),
                "both_hits@3": get_metric(metrics, "both.realistic.hits@3", "hits@3"),
                "both_hits@10": get_metric(metrics, "both.realistic.hits@10", "hits@10", "hits@k"),
                "head_hits@10": get_metric(metrics, "head.realistic.hits@10"),
                "tail_hits@10": get_metric(metrics, "tail.realistic.hits@10"),
                "entity_count": len(tf_train.entity_to_id),
                "relation_count": len(tf_train.relation_to_id),
            }
            results_rows.append(row)

            flat_metrics = metrics.to_flat_dict()
            (run_dir / "metrics_flat.json").write_text(json.dumps(flat_metrics, indent=2), encoding="utf-8")

            try:
                save_embeddings(run_dir, tf_train, result.model)
                entity_embeddings = np.load(run_dir / "entity_embeddings.npy")
                reference_entities = degree_ranked_entities(train)
                write_nearest_neighbors(run_dir, tf_train, entity_embeddings, reference_entities)
                plot_tsne(run_dir, tf_train, entity_embeddings, type_labels, args.tsne_sample_size, args.seed)
            except Exception as exc:
                (run_dir / "analysis_warning.txt").write_text(
                    f"Could not complete embedding analysis automatically: {exc}\n",
                    encoding="utf-8",
                )

            score = row["both_mrr"] if row["both_mrr"] is not None else -1.0
            if best_run is None or score > best_run["score"]:
                best_run = {
                    "score": score,
                    "size": len(subset),
                    "model": model_name,
                    "run_dir": str(run_dir),
                    "row": row,
                }

    summary_json = {
        "config": {
            "models": args.models,
            "sizes": size_requests,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_negs_per_pos": args.num_negs_per_pos,
            "seed": args.seed,
        },
        "results": results_rows,
        "best_run": best_run,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Saved summary JSON to {args.output_dir / 'summary.json'}")
    print(f"Saved summary CSV to {args.output_dir / 'summary.csv'}")
    print(json.dumps(summary_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
