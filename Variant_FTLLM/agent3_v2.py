import requests
import json

AGENT3_URL = "https://rectify-prevail-rectified.ngrok-free.dev"

def test_health():
    print("--- Health Check ---")
    try:
        r = requests.get(f"{AGENT3_URL}/health", timeout=5)
        print(f" Server is up: {r.json()}")
    except Exception as e:
        print(f"Server unreachable: {e}")
        return False
    return True

def test_generate(question, sparql_results):
    print(f"\nTest")
    print(f"Question: {question}")
    print(f"SPARQL Results: {json.dumps(sparql_results, indent=2)}")
    print("Calling Agent 3...")

    r = requests.post(
        f"{AGENT3_URL}/generate",
        json={"question": question, "sparql_results": json.dumps(sparql_results)},
        timeout=60
    )

    if r.status_code == 200:
        print(f"\n Answer:\n{r.json()['answer']}")
    else:
        print(f" Error {r.status_code}: {r.text}")


if __name__ == "__main__":
    if not test_health():
        exit()

    # Test 1
    test_generate(
        question="What is the diagnosis for this patient?",
        sparql_results={
            "diagnostic": "BCC",
            "diagnosis_full": "Basal Cell Carcinoma",
            "clinical_data": "Patient: 62-year-old male. Location: nose. Symptoms: bleeding, growth. Biopsy: True."
        }
    )

    # Test 2
    test_generate(
        question="What treatment options are relevant for this diagnosis?",
        sparql_results=[
            {"treatment": "Surgical excision", "type": "Primary"},
            {"treatment": "Mohs surgery", "type": "Alternative"}
        ]
    )

    # Test 3
    test_generate(
        question="What are the symptoms of melanoma?",
        sparql_results=[
            {"symptom": "asymmetric_lesion"},
            {"symptom": "irregular_border"},
            {"symptom": "color_variation"},
            {"symptom": "diameter_greater_than_6mm"}
        ]
    )