
from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from typing import List, Tuple
import requests
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

OLLAMA_URL = "http://localhost:11434/api/generate"

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

def ask_local_llm(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")

def load_graph(path: Path) -> Graph:
    g = Graph()
    fmt = "nt" if path.suffix.lower()==".nt" else "turtle"
    g.parse(path, format=fmt)
    logging.info("Loaded %s triples from %s", len(g), path)
    return g

def local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#",1)[1]
    return s.rsplit("/",1)[-1]

def build_schema_summary(g: Graph, out_path: Path) -> str:
    prefixes = []
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    defaults = {
        "rdf": str(RDF), "rdfs": str(RDFS), "owl": str(OWL),
    }
    for k,v in defaults.items():
        ns_map.setdefault(k,v)
    for p, ns in sorted(ns_map.items()):
        prefixes.append(f"PREFIX {p}: <{ns}>")
    preds = sorted({str(p) for _,p,_ in g})[:120]
    classes = sorted({str(o) for _,_,o in g.triples((None, RDF.type, None)) if isinstance(o, URIRef)})[:80]
    labels = sorted({str(p) for _,p,_ in g if p == RDFS.label})[:5]
    sample = []
    for i, (s,p,o) in enumerate(g):
        if i>=25: break
        sample.append(f"- {s} {p} {o}")
    summary = "\n".join(prefixes) + "\n# Predicates\n" + "\n".join(f"- {p}" for p in preds) + \
        "\n# Classes\n" + "\n".join(f"- {c}" for c in classes) + "\n# Sample triples\n" + "\n".join(sample)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    logging.info("Saved schema summary to %s", out_path)
    return summary

SPARQL_INSTRUCTIONS = """
You generate SPARQL SELECT queries for a LOCAL RDF graph loaded in rdflib.
Rules:
- Use ONLY prefixes/IRIs shown in the schema summary.
- Do NOT use Wikidata endpoint shortcuts like wdt:, wd:, p:, ps: unless they are explicitly in the schema summary.
- Prefer simple SELECT queries.
- Return ONLY one SPARQL query in a fenced ```sparql code block.
- No explanations.
"""

def make_prompt(schema_summary: str, question: str) -> str:
    return f"{SPARQL_INSTRUCTIONS}\nSCHEMA SUMMARY:\n{schema_summary}\nQUESTION:\n{question}\nReturn only the SPARQL query."

def extract_sparql(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    q = m.group(1).strip() if m else text.strip()
    q = q.strip()
    return q

BAD_PATTERNS = [
    re.compile(r"^\s*(Elaborate|Explain|Here(?:'s| is)|The query)", re.I),
]

def is_probably_sparql(query: str) -> bool:
    if any(p.search(query) for p in BAD_PATTERNS):
        return False
    return bool(re.match(r"^\s*(PREFIX\b.*\n\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", query, re.I|re.S))

def try_llm_query(question: str, schema_summary: str, model: str) -> str|None:
    raw = ask_local_llm(make_prompt(schema_summary, question), model)
    q = extract_sparql(raw)
    if is_probably_sparql(q):
        return q
    return None

def entity_fallback_queries(g: Graph, question: str) -> list[str]:
    q_lower = question.lower().strip().rstrip("?")
    phrase = None
    for prefix in ("who is ", "what is ", "tell me about "):
        if q_lower.startswith(prefix):
            phrase = question[len(prefix):].strip().rstrip("?")
            break
    if not phrase:
        phrase = question.strip().rstrip("?")
    escaped = phrase.replace('"', '\\"')
    queries = []
    queries.append(f"""
PREFIX rdfs: <{RDFS}>
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escaped}")))
  ?s ?p ?o .
}}
LIMIT 30
""".strip())
    # If graph has no labels, try local-name matching through regex on subject IRIs
    token = re.sub(r"[^A-Za-z0-9]+", " ", phrase).strip()
    regex = "|".join(re.escape(t) for t in token.split() if t)
    if regex:
        queries.append(f"""
SELECT ?s ?p ?o WHERE {{
  ?s ?p ?o .
  FILTER(REGEX(STR(?s), "{regex}", "i"))
}}
LIMIT 30
""".strip())
    return queries

def run_sparql(g: Graph, query: str) -> tuple[list[str], list[tuple[str,...]]]:
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(c) for c in row) for row in res]
    return vars_, rows

def answer_no_rag(question: str, model: str) -> str:
    return ask_local_llm(f"Answer briefly and directly:\n\n{question}", model)

def pretty_print_result(result: dict):
    print("\n[Repaired?]", result.get("repaired", False))
    print("\n[SPARQL Query]")
    print(result.get("query",""))
    if result.get("error"):
        print("\n[Execution Error]")
        print(result["error"])
    print("\n[Results]")
    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("[No rows returned]")
        return
    if vars_:
        print(" | ".join(vars_))
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")

def answer_with_rag(g: Graph, schema_summary: str, question: str, model: str) -> dict:
    # 1) try LLM-generated local query
    q = try_llm_query(question, schema_summary, model)
    attempted = []
    if q:
        attempted.append(q)
    # 2) fallback local label/IRI queries
    attempted.extend(entity_fallback_queries(g, question))
    seen = set()
    for idx, query in enumerate(attempted):
        if query in seen: 
            continue
        seen.add(query)
        try:
            vars_, rows = run_sparql(g, query)
            if rows:
                return {"query": query, "vars": vars_, "rows": rows, "repaired": idx>0, "error": None}
        except Exception as e:
            last_err = str(e)
            continue
    return {"query": attempted[0] if attempted else "", "vars": [], "rows": [], "repaired": len(attempted)>1, "error": "No valid SPARQL query could be produced."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gemma3")
    ap.add_argument("--schema-out", type=Path, default=Path("reports/lab6_schema_summary.txt"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    g = load_graph(args.graph)
    schema_summary = build_schema_summary(g, args.schema_out)
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        print("\n--- Baseline (No RAG) ---")
        try:
            print(answer_no_rag(q, args.model))
        except Exception as e:
            print(f"[Baseline error] {e}")
        print("\n--- SPARQL-generation RAG ---")
        result = answer_with_rag(g, schema_summary, q, args.model)
        pretty_print_result(result)

if __name__ == "__main__":
    main()
