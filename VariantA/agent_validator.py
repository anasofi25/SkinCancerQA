import ollama

MODEL = "llama3.2"


def is_wrong_type(question, results):
    """Rule-based check for obvious type mismatches between question intent
    and returned results — runs instantly without calling the LLM."""
    q = question.lower()
    result_words = [w.strip().lower() for w in results.split("\n") if w.strip()]

    diagnosis_terms = {"biopsy", "dermoscopy", "visualexamination"}
    treatment_terms = {"immunotherapy", "surgicalexcision", "radiationtherapy",
                       "chemotherapy", "cryotherapy", "mohssurgery", "topicaltherapy"}

    result_set = set(result_words)

    if "treat" in q and result_set & diagnosis_terms:
        return True

    if "diagnos" in q and "diagnosed as" not in q and result_set & treatment_terms:
        return True

    return False


def validate_results(user_question, sparql_query, formatted_results):
    """Two-layer validator: rule-based check first, then LLM-based
    semantic validation if the rules pass."""

    if is_wrong_type(user_question, formatted_results):
        return False, "Detected type mismatch (rule-based)", "RULE_FAIL"

    results_list = [
        r.strip() for r in formatted_results.split("\n") if r.strip()
    ]

    if not results_list or "no results found" in formatted_results.lower():
        return False, "Empty result set", "RULE_FAIL"

    prompt = f"""
Question: "{user_question}"
Results: {formatted_results}

The results are VALID if:
1. They are not empty
2. The result values make sense for the question

Examples of VALID cases:
- Question about symptoms → results contain: Asymmetry, Bleeding, ColorVariation etc. → VALID
- Question about treatments → results contain: SurgicalExcision, Immunotherapy, Chemotherapy etc. → VALID
- Question about which cancers have a property → results contain cancer names like Melanoma → VALID
- Question about risk factors → results contain: UVExposure, FamilyHistory etc. → VALID
- Question about body regions → results contain: Face, Arms, Neck etc. → VALID
- Question about cancer types → results contain: Melanoma, BasalCellCarcinoma etc. → VALID
- Question "which cancers are treated with X" → results contain CANCER NAMES → VALID
- Question "which cancers have risk factor X" → results contain CANCER NAMES → VALID

The results are INVALID only if:
1. Results are empty or say "No results found"
2. Results are clearly the WRONG TYPE (treatments returned for a diagnosis question)

IMPORTANT: If the question starts with "Which cancers..." then cancer names
in the results are ALWAYS correct regardless of what property is mentioned.

Reply ONLY with:
VALID: yes
or
VALID: no
REASON: one sentence
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a SPARQL query validator. You MUST respond with either 'VALID: yes' or 'VALID: no' followed by a REASON. Never ask for more information. Always validate based on what you are given."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response['message']['content'].strip()

    # Check only the first line to avoid false negatives from models that
    # add explanatory text after the verdict
    first_line = answer.split("\n")[0].lower()
    is_valid = "valid: yes" in first_line

    reason = ""
    if not is_valid and "reason:" in answer.lower():
        reason = answer.lower().split("reason:")[-1].strip()

    return is_valid, reason, answer


if __name__ == "__main__":
    # Regression tests covering the main validation scenarios
    tests = [
        {
            "name": "Test 1 - Correct results (should be VALID)",
            "question": "What are the symptoms of melanoma?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?symptom WHERE { sc:Melanoma sc:hasSymptom ?symptom }",
            "results": "Asymmetry\nBleeding\nColorVariation\nDiameterChange\nIrregularBorder",
            "expected": True
        },
        {
            "name": "Test 2 - Empty results (should be INVALID)",
            "question": "How is skin cancer diagnosed?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?method WHERE { sc:SkinCancer sc:diagnosedBy ?method }",
            "results": "No results found",
            "expected": False
        },
        {
            "name": "Test 3 - Wrong property used (should be INVALID)",
            "question": "How is melanoma treated?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?method WHERE { sc:Melanoma sc:diagnosedBy ?method }",
            "results": "Biopsy\nDermoscopy",
            "expected": False
        },
        {
            "name": "Test 4 - Correct reverse lookup (should be VALID)",
            "question": "Which cancers are treated with immunotherapy?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?cancer WHERE { ?cancer sc:treatedWith sc:Immunotherapy }",
            "results": "Melanoma",
            "expected": True
        },
        {
            "name": "Test 5 - Correct risk factor lookup (should be VALID)",
            "question": "Which cancers have UV exposure as a risk factor?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?cancer WHERE { ?cancer sc:hasRiskFactor sc:UVExposure }",
            "results": "Melanoma\nBasalCellCarcinoma\nSquamousCellCarcinoma\nActinicKeratosis",
            "expected": True
        },
        {
            "name": "Test 6 - Treatment question with correct results (should be VALID)",
            "question": "How is melanoma treated?",
            "sparql": "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\nSELECT ?treatment WHERE { sc:Melanoma sc:treatedWith ?treatment }",
            "results": "Chemotherapy\nImmunotherapy\nSurgicalExcision",
            "expected": True
        },
    ]

    passed = 0
    for test in tests:
        print(f"\n{test['name']}")
        is_valid, reason, raw = validate_results(
            test["question"], test["sparql"], test["results"]
        )
        ok = is_valid == test["expected"]
        passed += ok
        print(f"  Expected: {'VALID' if test['expected'] else 'INVALID'} | "
              f"Got: {'VALID' if is_valid else 'INVALID'} | "
              f"{'PASSED' if ok else 'FAILED'}")
        if reason:
            print(f"  Reason: {reason}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")