from __future__ import annotations

import argparse
import json
from pathlib import Path

import owlready2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 5 SWRL reasoning on family.owl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-ontology", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument("--base-iri", type=str, default=None)
    parser.add_argument("--java-exe", type=str, default=None)
    return parser


def load_ontology(input_path: Path):
    # Use plain local path; Owlready2 + Git Bash on Windows can choke on file:// URIs
    onto = owlready2.get_ontology(str(input_path)).load()
    return onto


def find_person_and_age(onto):
    person_cls = getattr(onto, "Person", None)
    if person_cls is None:
        # fallback: first class whose name resembles person
        for cls in onto.classes():
            if cls.name.lower() == "person" or "person" in cls.name.lower():
                person_cls = cls
                break
    if person_cls is None:
        raise ValueError("Could not find a Person class in family.owl")

    age_prop = getattr(onto, "age", None)
    if age_prop is None:
        for prop in onto.data_properties():
            if prop.name.lower() == "age":
                age_prop = prop
                break
    if age_prop is None:
        raise ValueError("Could not find an age data property in family.owl")

    return person_cls, age_prop


def ensure_old_person(onto, person_cls):
    old_person_cls = getattr(onto, "oldPerson", None)
    if old_person_cls is None:
        with onto:
            class oldPerson(person_cls):  # type: ignore[misc, valid-type]
                pass
        old_person_cls = onto.oldPerson
    return old_person_cls


def get_people(onto, person_cls):
    # Use all individuals if the ontology is small; then filter by instance check where possible
    people = []
    for ind in onto.individuals():
        try:
            if person_cls in ind.is_a or ind.is_instance_of(person_cls):
                people.append(ind)
                continue
        except Exception:
            pass
        # fallback by class membership ancestry
        try:
            if any(getattr(c, "name", "").lower() == getattr(person_cls, "name", "").lower() for c in ind.is_a if hasattr(c, "name")):
                people.append(ind)
        except Exception:
            pass
    # if none found, just return all individuals that have age values
    return people


def apply_manual_rule(people, age_prop, old_person_cls):
    inferred = []
    for ind in people:
        try:
            values = list(age_prop[ind])  # Owlready2 indexing syntax
        except Exception:
            values = list(getattr(ind, age_prop.name, []))
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except Exception:
                continue
        if numeric_values and max(numeric_values) > 60:
            if old_person_cls not in ind.is_a:
                ind.is_a.append(old_person_cls)
            inferred.append({"individual": ind.name, "age": max(numeric_values)})
    return inferred


def main() -> int:
    args = build_arg_parser().parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {input_path}")

    if args.java_exe:
        owlready2.JAVA_EXE = args.java_exe

    onto = load_ontology(input_path)
    with onto:
        person_cls, age_prop = find_person_and_age(onto)
        old_person_cls = ensure_old_person(onto, person_cls)

        # Create a SWRL rule object as required by the lab
        rule = owlready2.Imp()
        rule.set_as_rule(f"{person_cls.name}(?p), {age_prop.name}(?p, ?a), greaterThan(?a, 60) -> {old_person_cls.name}(?p)")

    reasoner_status = "pellet_success"
    reasoner_error = None
    inferred = []

    try:
        owlready2.sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
        # collect inferred individuals
        for ind in onto.individuals():
            if old_person_cls in getattr(ind, "is_a", []):
                age_vals = []
                try:
                    age_vals = list(age_prop[ind])
                except Exception:
                    age_vals = list(getattr(ind, age_prop.name, []))
                age_val = age_vals[0] if age_vals else None
                inferred.append({"individual": ind.name, "age": age_val})
    except Exception as e:
        # Fallback: Java/Pellet mismatch on Windows can break Owlready2; apply the SWRL logic manually
        reasoner_status = "manual_fallback"
        reasoner_error = str(e)
        inferred = apply_manual_rule(get_people(onto, person_cls), age_prop, old_person_cls)

    args.output_ontology.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)

    onto.save(file=str(args.output_ontology), format="rdfxml")

    report = {
        "input_ontology": str(input_path),
        "output_ontology": str(args.output_ontology),
        "person_class": person_cls.name,
        "age_property": age_prop.name,
        "rule": f"{person_cls.name}(?p), {age_prop.name}(?p, ?a), greaterThan(?a, 60) -> {old_person_cls.name}(?p)",
        "reasoner_status": reasoner_status,
        "reasoner_error": reasoner_error,
        "old_person_class": old_person_cls.name,
        "old_person_instances": inferred,
        "old_person_count": len(inferred),
    }

    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Lab 5 - family.owl SWRL reasoning",
        "",
        f"Input ontology: {input_path}",
        f"Output ontology: {args.output_ontology}",
        f"Person class: {person_cls.name}",
        f"Age property: {age_prop.name}",
        f"Rule: {report['rule']}",
        f"Reasoner status: {reasoner_status}",
    ]
    if reasoner_error:
        lines += ["", "Reasoner/Pellet error (fallback used):", reasoner_error]
    lines += ["", f"oldPerson instances found: {len(inferred)}"]
    for item in inferred:
        lines.append(f"- {item['individual']} (age={item['age']})")
    args.report_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved inferred ontology to {args.output_ontology}")
    print(f"Saved JSON report to {args.report_json}")
    print(f"Saved text report to {args.report_txt}")
    print(json.dumps({
        "reasoner_status": reasoner_status,
        "old_person_count": len(inferred),
        "output_ontology": str(args.output_ontology),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
