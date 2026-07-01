"""
debug_agents12.py

Tests Agent 1 + Agent 2 on a question with full debug output.
Shows exactly what SPARQL was generated, what GraphDB returned,
and why validation passed or failed.

Change QUESTION at the bottom to test different questions.
"""

from agent_gen_query import generate_sparql
from agent_validator import validate_results
import requests
import time

GRAPHDB_ENDPOINT = "http://localhost:7200/repositories/skin_cancer_expanded"
MAX_RETRIES = 2


def execute_query(query):
    headers = {"Accept": "application/sparql-results+json"}
    print(f"\n  [GraphDB] Sending query to {GRAPHDB_ENDPOINT}")
    start = time.perf_counter()
    try:
        response = requests.get(
            GRAPHDB_ENDPOINT,
            params={"query": query},
            headers=headers,
            timeout=10
        )
    except requests.exceptions.Timeout:
        print("  [GraphDB] ERROR: Timeout after 10s")
        return "Error: GraphDB timeout"
    except requests.exceptions.ConnectionError:
        print("  [GraphDB] ERROR: Cannot connect — is GraphDB running?")
        return "Error: GraphDB not running"

    elapsed = round(time.perf_counter() - start, 3)
    print(f"  [GraphDB] Response: HTTP {response.status_code} in {elapsed}s")

    if response.status_code != 200:
        print(f"  [GraphDB] ERROR body: {response.text[:300]}")
        return f"Error: {response.status_code}"

    data = response.json()

    if "results" not in data:
        print("  [GraphDB] ERROR: No 'results' key in response")
        print(f"  [GraphDB] Response keys: {list(data.keys())}")
        return "No results found"

    bindings = data["results"]["bindings"]
    print(f"  [GraphDB] Raw bindings count: {len(bindings)}")

    if not bindings:
        print("  [GraphDB] Zero bindings returned — query ran but matched nothing in ontology")
        return "No results found"

    results = []
    EXCLUDE = {"case01", "case02", "case03", "case04", "case05"}
    for binding in bindings:
        for var in binding:
            value = binding[var]["value"]
            raw_value = value
            if "#" in value:
                value = value.split("#")[-1]
            if value.lower() in EXCLUDE:
                print(f"  [GraphDB] Excluded: {raw_value}")
            else:
                results.append(value)

    print(f"  [GraphDB] Results after filtering: {results}")
    return "\n".join(results) if results else "No results found"


def debug_pipeline(question, max_retries=MAX_RETRIES):
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    feedback = None

    for attempt in range(max_retries + 1):
        print(f"\n{'─' * 70}")
        print(f"ATTEMPT {attempt + 1} / {max_retries + 1}")
        print(f"{'─' * 70}")

        if feedback:
            print(f"\n  [Feedback to Agent 1]: {feedback}")

        # ── Agent 1 ──────────────────────────────────────────────────────────
        print("\n  [Agent 1] Generating SPARQL...")
        start = time.perf_counter()
        sparql = generate_sparql(question, feedback)
        a1_time = round(time.perf_counter() - start, 2)
        print(f"  [Agent 1] Done in {a1_time}s")
        print(f"\n  [Agent 1] Generated SPARQL:\n{sparql}")

        # ── GraphDB ───────────────────────────────────────────────────────────
        results = execute_query(sparql)
        print(f"\n  [GraphDB] Final results string:\n  {repr(results)}")

        # ── Agent 2 ───────────────────────────────────────────────────────────
        print("\n  [Agent 2] Validating...")
        start = time.perf_counter()
        is_valid, reason, raw_response = validate_results(question, sparql, results)
        a2_time = round(time.perf_counter() - start, 2)

        print(f"  [Agent 2] Done in {a2_time}s")
        print(f"  [Agent 2] Valid: {is_valid}")
        if reason:
            print(f"  [Agent 2] Reason for INVALID: {reason}")
        print(f"  [Agent 2] Full LLM response:\n  {raw_response}")

        if is_valid:
            print(f"\n{'=' * 70}")
            print(f"SUCCESS at attempt {attempt + 1}")
            print(f"SPARQL: {sparql}")
            print(f"Results: {results}")
            print(f"{'=' * 70}")
            return sparql, results

        # diagnosis: why did it fail?
        print(f"\n  [DEBUG] Failure diagnosis:")
        if "No results found" in results or "Error:" in results:
            print("  → GraphDB returned empty/error — SPARQL query likely uses wrong property names,")
            print("    non-existent classes, or blank nodes instead of ontology class URIs.")
            print("    Check if the SPARQL uses sc:Itching vs 'itching' string literal.")
        elif results:
            print(f"  → GraphDB DID return results ({results[:100]}...)")
            print("    But Agent 2 rejected them — mismatch between results and question intent.")

        feedback = reason or "The query does not answer the question correctly."
        print(f"\n  [Retry] New feedback for Agent 1: '{feedback}'")

    print(f"\n{'=' * 70}")
    print(f"FAILED after {max_retries + 1} attempts")
    print(f"{'=' * 70}")
    return None, None


if __name__ == "__main__":
    #QUESTION = "A 62-year-old female smoker has an itchy, bleeding lesion on her chest diagnosed as squamous cell carcinoma. What are the available treatments?"
    #QUESTION = "A 68-year-old patient with years of sun exposure has a scaly, itchy patch on the forearm that has been slowly changing. What skin condition is this likely to be?"
    #QUESTION = "What is skin cancer and how dangerous is it?"
    #QUESTION = "A 78-year-old man who smokes and has had skin cancer before develops a lesion on his back. Which cancer has both smoking and previous skin cancer as risk factors?"
    #QUESTION = "A doctor is examining a suspicious mole using dermoscopy. What cancer is typically diagnosed this way?"
    #QUESTION = "A 77-year-old patient has a persistent rough patch on the face that won't go away. Without knowing the diagnosis, what treatments exist for precancerous skin lesions?"
    QUESTION = "A 49-year-old woman notices a mole on her thigh that has been growing, changing and becoming raised over several weeks. What is the most likely diagnosis and how is it confirmed?"

    debug_pipeline(QUESTION)