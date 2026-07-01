"""
run_all_testing_pipeline2.py

Runs ALL questions (standard + clinical case + edge case) through the full
pipeline for llama3.2, skincancer-llama, and medgemma-original — in that order.
Saves results to testing_pipeline2.csv with the same schema as testing_pipeline.csv,
plus an extra "debug_note" column with a short diagnosis when Agent 1/2 fails,
similar to the reasoning used in debug_agents12.py.
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

# ── All questions — exactly the 25 from case_questions.txt ───────────────────
QUESTIONS = [
    # Q1 — PAT_966 (46F, FACE, MEL)
    ("A 46-year-old woman has a mole on her face that has been itching and visibly changing color over the past few months. What type of skin cancer could this be?", "Melanoma"),
    # Q2 — PAT_680 (78M, BACK, MEL)
    ("A 78-year-old male smoker with a history of previous skin cancer has developed a growing lesion on his back that has changed in appearance. What skin cancer has UV exposure and previous skin cancer as risk factors?", "Melanoma"),
    # Q3 — PAT_46 (55F, NECK, BCC)
    ("A 55-year-old woman with a history of skin cancer presents with an itchy, bleeding, raised lesion on her neck. What type of skin cancer commonly appears on the neck?", "BasalCellCarcinoma"),
    # Q4 — PAT_684 (79M, FOREARM, BCC)
    ("A 79-year-old man with very fair skin has a raised, itchy lesion on his forearm that bleeds occasionally. What skin cancers are associated with fair skin as a risk factor?", "BasalCellCarcinoma"),
    # Q5 — PAT_778 (52F, FACE, BCC)
    ("A 52-year-old woman has a growing, bleeding, elevated lesion on her face confirmed to be basal cell carcinoma. What treatments are available for her?", "MohsSurgery"),
    # Q6 — PAT_714 (67M, SCC)
    ("A 67-year-old male smoker presents with an itchy, growing, raised lesion. Smoking is a known risk factor for which type of skin cancer?", "SquamousCellCarcinoma"),
    # Q7 — PAT_319 (62F, CHEST, SCC)
    ("A 62-year-old female smoker has an itchy, bleeding lesion on her chest diagnosed as squamous cell carcinoma. What are the available treatments?", "SurgicalExcision"),
    # Q8 — PAT_1545 (77, FACE, ACK)
    ("A 77-year-old patient has a rough, persistently itchy patch on their face that does not heal. What condition does this suggest and how is it typically treated?", "Cryotherapy"),
    # Q9 — PAT_1995 (68, FOREARM, ACK)
    ("A 68-year-old patient with years of sun exposure has a scaly, itchy patch on the forearm that has been slowly changing. What skin condition is this likely to be?", "ActinicKeratosis"),
    # Q10 — PAT_117 (74F, FACE, BCC)
    ("A 74-year-old woman with very fair skin has a lesion on her face that itches, bleeds and appears raised. Her doctor suspects basal cell carcinoma. What diagnostic method would confirm this?", "Biopsy"),

    # EDGE CASE 1: Region NOT in ontology (EAR)
    ("A 67-year-old male smoker has a lesion on his ear. What skin cancers commonly appear on the ear?", "SquamousCellCarcinoma"),
    # EDGE CASE 2: Region NOT in ontology (FOREARM)
    ("A 68-year-old patient has a lesion on the forearm. What skin cancers are commonly found on the forearm?", "ActinicKeratosis"),
    # EDGE CASE 3: Diagnostic NOT in ontology (NEV = Nevus)
    ("A 23-year-old patient has a pigmented lesion on the face that has not changed. Could this be a nevus and is it dangerous?", "Melanoma"),
    # EDGE CASE 4: Negation — smoking NOT linked to BCC
    ("A 73-year-old woman has basal cell carcinoma on her face. Is smoking a risk factor for this condition?", "UVExposure"),
    # EDGE CASE 5: Symptom shared by multiple cancers
    ("A patient has a lesion that bleeds. Which skin cancers have bleeding as a symptom?", "Melanoma"),
    # EDGE CASE 6: Too vague — no specific entity
    ("What is skin cancer and how dangerous is it?", "Melanoma"),
    # EDGE CASE 7: Multiple risk factors combined
    ("A 78-year-old man who smokes and has had skin cancer before develops a lesion on his back. Which cancer has both smoking and previous skin cancer as risk factors?", "Melanoma"),
    # EDGE CASE 8: Symptoms + region combination
    ("A 55-year-old woman has an itchy, bleeding, raised lesion on her neck. What skin cancer appears on the neck with these symptoms?", "BasalCellCarcinoma"),
    # EDGE CASE 9: Ambiguous forward vs reverse
    ("A doctor is examining a suspicious mole using dermoscopy. What cancer is typically diagnosed this way?", "Melanoma"),
    # EDGE CASE 10: Region partially in ontology (CHEST not in ontology)
    ("A 62-year-old female smoker has a lesion on her chest. Which skin cancers are associated with smoking?", "SquamousCellCarcinoma"),
    # EDGE CASE 11: Treatment question with no diagnosis given
    ("A 77-year-old patient has a persistent rough patch on the face that won't go away. Without knowing the diagnosis, what treatments exist for precancerous skin lesions?", "Cryotherapy"),
    # EDGE CASE 12: Diagnostic not in ontology (SEK = Seborrheic Keratosis)
    ("A 65-year-old patient has a rough, dark lesion on the face that looks like it could be cancerous. How would a doctor confirm whether this is malignant?", "Biopsy"),
    # EDGE CASE 13: Mixed cancer history + new symptoms
    ("A 49-year-old woman notices a mole on her thigh that has been growing, changing and becoming raised over several weeks. What is the most likely diagnosis and how is it confirmed?", "Biopsy"),
    # EDGE CASE 14: Survival rate / data property not SPARQL-queryable
    ("What is the survival rate for melanoma?", "Melanoma"),
    # EDGE CASE 15: Reverse lookup with uncommon treatment
    ("A surgeon is planning Mohs surgery for a patient. Which skin cancer is most commonly treated this way?", "BasalCellCarcinoma"),
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
    """
    Mirrors the debug logic from debug_agents12.py — produces a short
    human-readable explanation of why the Agent1/2 step failed.
    """
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

    # Agent 3
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


# ── Main: model-by-model, question-by-question ────────────────────────────────
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