import ollama

MODEL = "llama3.2"

def explain_answer(question, results):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant explaining skin cancer information clearly and concisely to students and medical professionals. Never mention databases, ontologies, SPARQL or technical terms."
            },
            {
                "role": "user",
                "content": f"""
A user asked: "{question}"

The medical knowledge base returned: {results}

Explain the answer in 2-3 clear sentences.
- Use simple medical language
- Use the exact names from the results
- Do NOT mention ontologies, SPARQL, databases or technical terms
- Just answer naturally as a medical assistant would
"""
            }
        ]
    )
    return response['message']['content'].strip()


if __name__ == "__main__":

    tests = [
        {
            "question": "What are the symptoms of melanoma?",
            "results": "Asymmetry\nBleeding\nColorVariation\nDiameterChange\nIrregularBorder"
        },
        {
            "question": "How is melanoma treated?",
            "results": "Chemotherapy\nImmunotherapy\nSurgicalExcision"
        },
        {
            "question": "Which cancers have UV exposure as a risk factor?",
            "results": "ActinicKeratosis\nBasalCellCarcinoma\nMelanoma\nSquamousCellCarcinoma"
        },
        {
            "question": "Where does basal cell carcinoma commonly appear?",
            "results": "Face\nNeck\nScalp"
        },
        {
            "question": "How is basal cell carcinoma diagnosed?",
            "results": "Biopsy\nVisualExamination"
        },
    ]

    for test in tests:
        print("\n" + "=" * 60)
        print(f"Question: {test['question']}")
        print(f"Results: {test['results']}")
        print("\nAnswer:")
        answer = explain_answer(test["question"], test["results"])
        print(answer)

print("\nDone")