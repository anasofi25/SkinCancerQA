"""
run_basic_questions.py

Runs the 15 basic ontology questions through the full pipeline for
llama3.2, skincancer-llama, and medgemma-original.
Appends results to testing_pipeline2.csv (same schema as before).
"""

from agent_gen_query import generate_sparql
from agent_validator import validate_results
from agent_explainer2 import explain_answer
import requests
import time
import csv
from pathlib import Path
from datetime import datetime

GRAPHDB_ENDPOINT = "http://localhost:7200/repositories/skin_cancer_expanded"

MODEL_REGISTRY = {
    #"llama3.2":          ("ollama", "llama3.2"),
   # "skincancer-llama":  ("ollama", "skincancer-llama"),
   # "medgemma-original": ("ollama", "medgemma"),
"medgemma-skincancer": ("lmstudio", "medgemma-skincancer"),
}

CSV_PATH = Path("../Results/testing_pipeline2.csv")

FIELDNAMES = [
    "timestamp",
    "question",
    "expected_keyword",
    "model",
    "valid",
    "retries",
    "agent1_time",
    "graphdb_time",
    "agent2_time",
    "agent3_time",
    "total_time",
    "keyword_hit",
    "answer_length_words",
    "sparql",
    "raw_results",
    "answer",
    "debug_note",
]

# ── 15 basic ontology questions ───────────────────────────────────────────────
QUESTIONS = [
    # BASIC 1
    ("What are the symptoms of melanoma?", "Asymmetry"),
    # BASIC 2
    ("What are the symptoms of basal cell carcinoma?", "PearlAppearance"),
    # BASIC 3
    ("How is melanoma treated?", "Immunotherapy"),
    # BASIC 4
    ("How is basal cell carcinoma treated?", "MohsSurgery"),
    # BASIC 5
    ("How is squamous cell carcinoma treated?", "SurgicalExcision"),
    # BASIC 6
    ("What are the risk factors for melanoma?", "UVExposure"),
    # BASIC 7
    ("What are the risk factors for skin cancer?", "UVExposure"),
    # BASIC 8
    ("What types of skin cancer exist?", "Melanoma"),
    # BASIC 9
    ("How is melanoma diagnosed?", "Biopsy"),
    # BASIC 10
    ("How is skin cancer diagnosed?", "Biopsy"),
    # BASIC 11
    ("Where does melanoma commonly appear?", "Back"),
    # BASIC 12
    ("Where does basal cell carcinoma commonly appear?", "Face"),
    # BASIC 13
    ("What symptoms does squamous cell carcinoma cause?", "ScalyPatch"),
    # BASIC 14
    ("What are the treatments for actinic keratosis?", "Cryotherapy"),
    # BASIC 15
    ("What are the risk factors for squamous cell carcinoma?", "Smoking"),
]


# ── GraphDB ───────────────────────────────────────────────────────────────────
def execute_query(query):
    headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.get(
            GRAPHDB_ENDPOINT,
            params={"query": query},
            headers=headers,
            timeout=10
        )
    except requests.exceptions.Timeout:
        return "Error: GraphDB timeout"
    except requests.exceptions.ConnectionError:
        return "Error: GraphDB not running"

    if response.status_code != 200:
        return f"Error: {response.status_code}"
    data = response.json()
    if "results" not in data:
        return "No results found"

    results = []
    EXCLUDE = {"case01", "case02", "case03", "case04", "case05"}
    for binding in data["results"]["bindings"]:
        for var in binding:
            value = binding[var]["value"]
            if "#" in value:
                value = value.split("#")[-1]
            if value.lower() not in EXCLUDE:
                results.append(value)

    if not results:
        return "No results found"
    return "\n".join(results)


def diagnose_failure(sparql: str, results: str, reason: str) -> str:
    if "No results found" in results or "Error:" in results:
        if "Region not in ontology" in reason:
            return reason
        return (
            "GraphDB returned empty/error — SPARQL likely references a class or "
            "region not present in the ontology (e.g. body region, diagnostic, or "
            "risk-factor combination not modeled), or uses blank nodes/string "
            "literals instead of ontology class URIs."
        )
    elif results:
        return (
            f"GraphDB DID return results ({results[:80]}...) but Agent 2 rejected "
            f"them — mismatch between results and question intent (reason: {reason})."
        )
    return reason or "Unknown failure"


# ── Agent 1 + Agent 2 ─────────────────────────────────────────────────────────
def get_sparql_and_results(question, max_retries=2):
    feedback = None
    total_agent1_time = 0
    total_agent2_time = 0
    graphdb_time = 0
    sparql_query = ""
    results = ""
    last_reason = None

    for attempt in range(max_retries + 1):
        print(f"    Attempt {attempt + 1}/{max_retries + 1}")

        start = time.perf_counter()
        sparql_query = generate_sparql(question, feedback)
        total_agent1_time += time.perf_counter() - start

        start = time.perf_counter()
        results = execute_query(sparql_query)
        graphdb_time += time.perf_counter() - start

        start = time.perf_counter()
        is_valid, reason, raw = validate_results(question, sparql_query, results)
        total_agent2_time += time.perf_counter() - start

        print(f"    Valid: {is_valid}" + (f" | {reason}" if reason else ""))

        if is_valid:
            return {
                "sparql": sparql_query,
                "results": results,
                "valid": True,
                "reason": None,
                "retries": attempt,
                "agent1_time": round(total_agent1_time, 2),
                "graphdb_time": round(graphdb_time, 4),
                "agent2_time": round(total_agent2_time, 2),
                "debug_note": "",
            }

        last_reason = reason
        feedback = reason or "The query does not answer the question correctly."

    debug_note = diagnose_failure(sparql_query, results, last_reason or "")

    return {
        "sparql": sparql_query,
        "results": results,
        "valid": False,
        "reason": last_reason,
        "retries": max_retries,
        "agent1_time": round(total_agent1_time, 2),
        "graphdb_time": round(graphdb_time, 4),
        "agent2_time": round(total_agent2_time, 2),
        "debug_note": debug_note,
    }


# ── CSV ───────────────────────────────────────────────────────────────────────
def save_row(row: dict):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Test a single question for a single model ─────────────────────────────────
def test_question_for_model(question: str, expected: str, model_key: str, max_retries: int = 2):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_start = time.perf_counter()

    print(f"\n  MODEL: {model_key}")

    try:
        run_data = get_sparql_and_results(question, max_retries)
    except Exception as e:
        print(f"  ERROR Agent1/2: {e}")
        save_row({
            "timestamp": timestamp,
            "question": question,
            "expected_keyword": expected,
            "model": model_key,
            "valid": False,
            "retries": max_retries,
            "agent1_time": None,
            "graphdb_time": None,
            "agent2_time": None,
            "agent3_time": None,
            "total_time": None,
            "keyword_hit": False,
            "answer_length_words": 0,
            "sparql": "",
            "raw_results": f"ERROR: {e}",
            "answer": f"ERROR: {e}",
            "debug_note": f"Exception during Agent1/2: {e}",
        })
        return

    sparql = run_data["sparql"]
    raw_results = run_data["results"]
    is_valid = run_data["valid"]

    if not is_valid:
        print(f"  Validation failed — skipping Agent 3")
        print(f"  [DEBUG] {run_data['debug_note']}")
        save_row({
            "timestamp": timestamp,
            "question": question,
            "expected_keyword": expected,
            "model": model_key,
            "valid": False,
            "retries": run_data["retries"],
            "agent1_time": run_data["agent1_time"],
            "graphdb_time": run_data["graphdb_time"],
            "agent2_time": run_data["agent2_time"],
            "agent3_time": None,
            "total_time": round(time.perf_counter() - pipeline_start, 2),
            "keyword_hit": False,
            "answer_length_words": 0,
            "sparql": sparql,
            "raw_results": raw_results,
            "answer": "FAILED - validation",
            "debug_note": run_data["debug_note"],
        })
        return

    agent3_start = time.perf_counter()
    try:
        answer = explain_answer(question, raw_results, model_key=model_key)
        agent3_time = round(time.perf_counter() - agent3_start, 2)
        print(f"  Agent 3 done in {agent3_time}s")
        print(f"  Answer: {answer[:120]}...")
    except Exception as e:
        answer = f"ERROR: {e}"
        agent3_time = None
        print(f"  ERROR Agent3: {e}")

    total_time = round(time.perf_counter() - pipeline_start, 2)
    keyword_hit = (
        expected.lower() in answer.lower()
        or expected.lower() in raw_results.lower()
    )
    answer_length = len(answer.split()) if answer and not answer.startswith("ERROR") else 0

    save_row({
        "timestamp": timestamp,
        "question": question,
        "expected_keyword": expected,
        "model": model_key,
        "valid": is_valid,
        "retries": run_data["retries"],
        "agent1_time": run_data["agent1_time"],
        "graphdb_time": run_data["graphdb_time"],
        "agent2_time": run_data["agent2_time"],
        "agent3_time": agent3_time,
        "total_time": total_time,
        "keyword_hit": keyword_hit,
        "answer_length_words": answer_length,
        "sparql": sparql,
        "raw_results": raw_results,
        "answer": answer,
        "debug_note": "",
    })

    print(f"  keyword_hit={keyword_hit} | total={total_time}s")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_all():
    total = len(QUESTIONS)
    model_order = list(MODEL_REGISTRY.keys())

    for model_key in model_order:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key}  ({total} questions)")
        print(f"{'#'*70}")

        for idx, (question, expected) in enumerate(QUESTIONS, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{total}] {question}")
            print(f"Expected: {expected}")
            print(f"{'='*70}")
            test_question_for_model(question, expected, model_key)

    print(f"\nAll results saved to: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    run_all()