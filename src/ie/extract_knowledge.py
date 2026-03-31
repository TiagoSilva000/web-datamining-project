"""Lab 1 information extraction.

Reads crawler_output.jsonl, runs spaCy NER + dependency parsing, and writes:
  - extracted_knowledge.csv (entity inventory)
  - relation_candidates.csv (candidate triples)

Example:
    python src/ie/extract_knowledge.py \
        --input data/processed/crawler_output.jsonl \
        --entity-output data/processed/extracted_knowledge.csv \
        --relation-output data/processed/relation_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


ALLOWED_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART", "EVENT"}
FALLBACK_MODELS = ["en_core_web_trf", "en_core_web_sm"]
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EntityRow:
    entity_text: str
    entity_normalized: str
    entity_type: str
    source_url: str
    mention_count: int
    example_sentence: str


@dataclass(frozen=True)
class RelationRow:
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    source_url: str
    sentence: str
    confidence: float
    method: str


def load_spacy_model(preferred: Optional[str] = None) -> Language:
    candidates = [preferred] if preferred else []
    candidates.extend(model for model in FALLBACK_MODELS if model != preferred)

    for model in candidates:
        if not model:
            continue
        try:
            return spacy.load(model)
        except OSError:
            continue

    raise OSError(
        "No spaCy English pipeline found. Install one with: "
        "python -m spacy download en_core_web_trf"
    )


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def normalize_text(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text.strip(" \t\n\r\"'“”‘’.,;:!?()[]{}")


def valid_entity_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    if len(text) == 1:
        return False
    if text.lower() in {"film", "series", "television", "movie", "tv", "season", "episode"}:
        return False
    return True


def entity_lookup(sentence: Span) -> Dict[int, Span]:
    lookup: Dict[int, Span] = {}
    for ent in sentence.ents:
        for i in range(ent.start, ent.end):
            lookup[i] = ent
    return lookup


def find_entity_for_token(token: Token, lookup: Dict[int, Span]) -> Optional[Span]:
    if token.i in lookup:
        return lookup[token.i]
    for ancestor in token.ancestors:
        if ancestor.i in lookup:
            return lookup[ancestor.i]
    return None


def predicate_from_verb(verb: Token, object_token: Optional[Token] = None) -> str:
    relation = verb.lemma_.lower().strip()
    if object_token is not None and object_token.dep_ == "pobj" and object_token.head.pos_ == "ADP":
        prep = object_token.head.lemma_.lower().strip()
        if prep:
            relation = f"{relation}_{prep}"
    return relation or verb.text.lower()


def extract_dependency_relations(sentence: Span) -> List[RelationRow]:
    lookup = entity_lookup(sentence)
    rows: List[RelationRow] = []
    url = sentence.doc.user_data.get("source_url", "")

    for token in sentence:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subject_entities: List[Span] = []
        object_entities: List[Tuple[Span, Token]] = []

        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                ent = find_entity_for_token(child, lookup)
                if ent is not None:
                    subject_entities.append(ent)
            elif child.dep_ in {"dobj", "obj", "attr", "dative", "oprd"}:
                ent = find_entity_for_token(child, lookup)
                if ent is not None:
                    object_entities.append((ent, child))
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        ent = find_entity_for_token(grandchild, lookup)
                        if ent is not None:
                            object_entities.append((ent, grandchild))

        for subj in subject_entities:
            for obj, obj_token in object_entities:
                subj_text = normalize_text(subj.text)
                obj_text = normalize_text(obj.text)
                if not valid_entity_text(subj_text) or not valid_entity_text(obj_text):
                    continue
                if subj_text == obj_text:
                    continue
                rows.append(
                    RelationRow(
                        subject=subj_text,
                        subject_type=subj.label_,
                        predicate=predicate_from_verb(token, obj_token),
                        object=obj_text,
                        object_type=obj.label_,
                        source_url=url,
                        sentence=normalize_text(sentence.text),
                        confidence=0.85,
                        method="dependency",
                    )
                )
    return rows


def extract_fallback_relations(sentence: Span) -> List[RelationRow]:
    ents = [ent for ent in sentence.ents if ent.label_ in ALLOWED_ENTITY_LABELS and valid_entity_text(ent.text)]
    if len(ents) < 2:
        return []

    root = sentence.root
    if root.pos_ not in {"VERB", "AUX", "NOUN", "PROPN", "ADJ"}:
        return []

    predicate = normalize_text(root.lemma_.lower() if root.lemma_ else root.text.lower())
    if not predicate:
        return []

    url = sentence.doc.user_data.get("source_url", "")
    rows: List[RelationRow] = []
    for left, right in zip(ents, ents[1:]):
        subj_text = normalize_text(left.text)
        obj_text = normalize_text(right.text)
        if subj_text == obj_text:
            continue
        if abs(right.start - left.end) > 15:
            continue
        rows.append(
            RelationRow(
                subject=subj_text,
                subject_type=left.label_,
                predicate=predicate,
                object=obj_text,
                object_type=right.label_,
                source_url=url,
                sentence=normalize_text(sentence.text),
                confidence=0.45,
                method="same_sentence_root",
            )
        )
    return rows


def dedupe_relations(rows: Iterable[RelationRow]) -> List[RelationRow]:
    best: Dict[Tuple[str, str, str, str], RelationRow] = {}
    for row in rows:
        key = (row.source_url, row.subject, row.predicate, row.object)
        current = best.get(key)
        if current is None or row.confidence > current.confidence:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r.source_url, r.subject, r.predicate, r.object))


def process_records(records: List[dict], nlp: Language) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_counts: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    entity_examples: Dict[Tuple[str, str, str], str] = {}
    relation_rows: List[RelationRow] = []

    texts = [record.get("text", "") for record in records]
    for record, doc in zip(records, nlp.pipe(texts, batch_size=4), strict=False):
        doc.user_data["source_url"] = record.get("url", "")
        for ent in doc.ents:
            if ent.label_ not in ALLOWED_ENTITY_LABELS:
                continue
            ent_text = normalize_text(ent.text)
            if not valid_entity_text(ent_text):
                continue
            key = (record.get("url", ""), ent_text, ent.label_)
            entity_counts[key]["mentions"] += 1
            entity_examples.setdefault(key, normalize_text(ent.sent.text))

        for sent in doc.sents:
            relation_rows.extend(extract_dependency_relations(sent))
            relation_rows.extend(extract_fallback_relations(sent))

    entities_output = [
        EntityRow(
            entity_text=key[1],
            entity_normalized=key[1],
            entity_type=key[2],
            source_url=key[0],
            mention_count=counter["mentions"],
            example_sentence=entity_examples[key],
        )
        for key, counter in entity_counts.items()
    ]
    entities_df = pd.DataFrame([row.__dict__ for row in sorted(entities_output, key=lambda r: (r.source_url, r.entity_type, r.entity_text))])

    relation_df = pd.DataFrame([row.__dict__ for row in dedupe_relations(relation_rows)])
    if not relation_df.empty:
        relation_df = relation_df.sort_values(["source_url", "subject", "predicate", "object"]).reset_index(drop=True)

    return entities_df, relation_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NER + relation candidates for Lab 1")
    parser.add_argument("--input", type=Path, required=True, help="crawler_output.jsonl")
    parser.add_argument("--entity-output", type=Path, required=True, help="CSV file for extracted entities")
    parser.add_argument("--relation-output", type=Path, required=True, help="CSV file for relation candidates")
    parser.add_argument("--model", type=str, default="en_core_web_trf", help="spaCy model to use")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        records = read_jsonl(args.input)
        nlp = load_spacy_model(args.model)
        entities_df, relations_df = process_records(records, nlp)
        args.entity_output.parent.mkdir(parents=True, exist_ok=True)
        args.relation_output.parent.mkdir(parents=True, exist_ok=True)
        entities_df.to_csv(args.entity_output, index=False, quoting=csv.QUOTE_MINIMAL)
        relations_df.to_csv(args.relation_output, index=False, quoting=csv.QUOTE_MINIMAL)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved entities to {args.entity_output}")
    print(f"Saved relation candidates to {args.relation_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
