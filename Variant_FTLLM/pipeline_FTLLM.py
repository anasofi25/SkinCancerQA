from VariantA.agent_gen_query import generate_sparql
from VariantA.agent_validator import validate_results
from ag3 import explain_answer
import requests
import threading

GRAPHDB_ENDPOINT = "http://localhost:7200/repositories/skin_cancer_expanded"

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


def run_pipeline(question, max_retries=2):
    feedback = None

    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1} ---")

        sparql_query = generate_sparql(question, feedback)
        print("SPARQL:\n", sparql_query)

        results = execute_query(sparql_query)
        print("Results:\n", results)

        is_valid, reason, raw = validate_results(question, sparql_query, results)
        print(f"Validation: {'VALID' if is_valid else 'INVALID'}")
        if reason:
            print(f"Reason: {reason}")

        if is_valid:
            answer = explain_answer(question, results)
            return sparql_query, results, answer

        feedback = reason or "The query does not answer the question correctly."

    return None, None, None


def run_pipeline_with_timeout(question, max_retries=2, timeout_seconds=120):
    result = [None, None, None]

    def target():
        r = run_pipeline(question, max_retries)
        result[0], result[1], result[2] = r

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"Timeout after {timeout_seconds}s")
        return None, None, None

    return result[0], result[1], result[2]


if __name__ == "__main__":
    questions = [
        "How is melanoma treated?",
        "Which cancers are treated with immunotherapy?",
        "Which cancers have UV exposure as a risk factor?"
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print("QUESTION:", q)
        sparql, results, answer = run_pipeline_with_timeout(q)
        if sparql:
            print("\nFINAL ANSWER:")
            print(answer)
        else:
            print("\nFAILED AFTER RETRIES")