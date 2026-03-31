# Web Datamining Project

Final project for Web Mining and Semantics.

This repository implements a full pipeline for:
1. Web crawling, cleaning, and information extraction
2. Knowledge base construction, alignment, and expansion
3. Rule-based reasoning and knowledge graph embeddings
4. RAG over RDF/SPARQL with a local LLM

## Repository Structure

src/
- crawl/    crawling + cleaning
- ie/       entity extraction + relation candidates
- kg/       KB construction, alignment, expansion, cleaning
- reason/   SWRL / family.owl reasoning
- kge/      split preparation + embedding training/evaluation
- rag/      local RAG over RDF/SPARQL

Other important folders:
- data/
- kg_artifacts/
- reports/
- notebooks/

## Topic / Domain

This project uses a movies and TV domain.

## Installation

Recommended: Python 3.10+ or Python 3.12

Install dependencies:
py -m pip install -r requirements.txt
py -m spacy download en_core_web_trf

## Ollama

Lab 6 uses a local LLM through Ollama.

Install Ollama on Windows, then run:
ollama run gemma3

Keep Ollama available locally while running the RAG scripts.

## Hardware Requirements

Minimum practical setup:
- Windows 10/11
- Python 3.10+
- Java installed if you want to try full Owlready2/Pellet reasoning
- At least 8 GB RAM recommended

Notes:
- KGE training was run on CPU
- Ollama/Gemma3 runs locally and may be slower on lower-end machines

## How to Run Each Module

### 1. Crawling + Cleaning
Input URLs are stored in:
data/samples/seed_urls_movies_tv.txt

Run:
py src/crawl/crawl_and_clean.py --input data/samples/seed_urls_movies_tv.txt --output data/crawler_output.jsonl --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36" --delay 2.0 --min-words 250

### 2. Information Extraction
Run:
py src/ie/extract_knowledge.py --input data/crawler_output.jsonl --entity-output data/extracted_knowledge.csv --relation-output data/relation_candidates.csv

### 3. Initial KB Construction
Run:
py src/kg/build_initial_kb.py --entities data/extracted_knowledge.csv --relations data/relation_candidates.csv --initial-kb kg_artifacts/initial_kb.ttl --ontology kg_artifacts/ontology.ttl --stats kg_artifacts/kb_stats_initial.json --min-relation-confidence 0.50

### 4. Entity / Predicate Alignment
Run:
py src/kg/align_wikidata.py --entities data/extracted_knowledge.csv --relations data/relation_candidates.csv --entity-links-out kg_artifacts/entity_links.csv --predicate-links-out kg_artifacts/predicate_alignment.csv --alignment-out kg_artifacts/alignment.ttl --min-entity-confidence 0.72 --min-property-confidence 0.75 --delay 0.15

### 5. KB Expansion
Run:
py src/kg/expand_kb.py --initial-kb kg_artifacts/initial_kb.ttl --alignment kg_artifacts/alignment.ttl --entity-links kg_artifacts/entity_links.csv --output kg_artifacts/expanded.nt --stats kg_artifacts/kb_stats_expanded.json --target-triples 50000 --max-depth 2 --per-entity-limit 250 --batch-size 10 --delay 0.20

### 6. Cleaning the Expanded Graph for KGE
Run:
py src/kg/clean_for_kge.py --input kg_artifacts/expanded.nt --output kg_artifacts/expanded_clean.nt --stats kg_artifacts/kb_stats_clean.json --largest-component

### 7. SWRL Reasoning on family.owl
Run:
py src/reason/reason_family_swrl.py --input family.owl --output-ontology kg_artifacts/family_inferred.owl --report-json reports/lab5_family_reasoning.json --report-txt reports/lab5_family_reasoning.txt

### 8. Prepare KGE Splits
Run:
py src/kge/prepare_kge_splits.py --input kg_artifacts/expanded_clean.nt --train-out data/kge/train.txt --valid-out data/kge/valid.txt --test-out data/kge/test.txt --stats-out kg_artifacts/kge_split_stats.json

### 9. Train and Evaluate KGE Models
Run:
py src/kge/train_kge_models.py --train data/kge/train.txt --valid data/kge/valid.txt --test data/kge/test.txt --graph-file kg_artifacts/expanded_clean.nt --output-dir reports/kge_runs --models TransE ComplEx --sizes 20000 50000 0 --embedding-dim 100 --epochs 20 --batch-size 256 --learning-rate 0.001 --num-negs-per-pos 10

## How to Run the RAG Demo

Start Ollama in PowerShell:
ollama run gemma3

Then in Git Bash run:
py src/rag/rag_sparql_chat.py --graph kg_artifacts/expanded_clean.nt --model gemma3

Evaluation:
py src/rag/evaluate_rag.py --graph kg_artifacts/expanded_clean.nt --questions data/samples/lab6_questions_movies_tv.txt --model gemma3 --csv-out reports/lab6_eval.csv --json-out reports/lab6_eval.json

## Main Outputs

### Lab 1
- data/crawler_output.jsonl
- data/extracted_knowledge.csv
- data/relation_candidates.csv

### Lab 4
- kg_artifacts/initial_kb.ttl
- kg_artifacts/ontology.ttl
- kg_artifacts/alignment.ttl
- kg_artifacts/expanded.nt
- kg_artifacts/entity_links.csv
- kg_artifacts/predicate_alignment.csv
- kg_artifacts/kb_stats_initial.json
- kg_artifacts/kb_stats_expanded.json

### Lab 5
- kg_artifacts/family_inferred.owl
- reports/lab5_family_reasoning.json
- reports/lab5_family_reasoning.txt
- data/kge/train.txt
- data/kge/valid.txt
- data/kge/test.txt
- kg_artifacts/kge_split_stats.json
- reports/kge_runs/summary.json
- reports/kge_runs/summary.csv

### Lab 6
- reports/lab6_schema_summary.txt
- reports/lab6_eval.csv
- reports/lab6_eval.json

## Screenshots

The repository includes two example screenshots in:

- `reports/screenshots/rag_demo.png`
- `reports/screenshots/kge_results.png`

### 1. RAG demo screenshot
`rag_demo.png` shows the interactive RDF/SPARQL chatbot running on the local knowledge graph with:
- a natural-language question
- the baseline answer without RAG
- the grounded SPARQL-based result from the local graph

### 2. KGE results screenshot
`kge_results.png` shows the summary table of KGE experiments, including:
- TransE and ComplEx
- 20k and full-dataset settings
- MRR and Hits@1/3/10 metrics

## Notes

- The project was developed incrementally from the lab sessions.
- Some reasoning behavior may depend on the local Java/Pellet setup.
- KGE and RAG quality depend heavily on extraction noise and alignment quality.
