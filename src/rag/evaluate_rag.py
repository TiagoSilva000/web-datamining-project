
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from rag_sparql_chat import load_graph, build_schema_summary, answer_no_rag, answer_with_rag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gemma3")
    ap.add_argument("--csv-out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    g = load_graph(args.graph)
    schema = build_schema_summary(g, Path("reports/lab6_schema_summary.txt"))
    questions = [line.strip() for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]

    rows = []
    for q in questions:
        try:
            baseline = answer_no_rag(q, args.model)
        except Exception as e:
            baseline = f"[Baseline error] {e}"
        rag = answer_with_rag(g, schema, q, args.model)
        rows.append({
            "question": q,
            "baseline": baseline,
            "rag_query": rag.get("query",""),
            "rag_repaired": rag.get("repaired", False),
            "rag_error": rag.get("error"),
            "rag_result_count": len(rag.get("rows", [])),
            "rag_first_row": " | ".join(rag["rows"][0]) if rag.get("rows") else "",
        })

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["question"])
        writer.writeheader()
        writer.writerows(rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved CSV to {args.csv_out}")
    print(f"Saved JSON to {args.json_out}")

if __name__ == "__main__":
    main()
