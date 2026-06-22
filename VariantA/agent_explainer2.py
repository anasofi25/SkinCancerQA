"""from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL = "medgemma-skincancer"

def explain_answer(question, results):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant explaining skin cancer information clearly and concisely to students and medical professionals. Never mention databases, ontologies, SPARQL or technical terms."
            },
            {
                "role": "user",
                "content": f
A user asked: "{question}"

The medical knowledge base returned: {results}

Explain the answer in 2-3 clear sentences.
- Use simple medical language
- Use the exact names from the results
- Do NOT mention ontologies, SPARQL, databases or technical terms
- Just answer naturally as a medical assistant would

            }
        ]
    )
    return response.choices[0].message.content.strip()

"""
import ollama
from openai import OpenAI

MODEL_REGISTRY = {
    "llama3.2": ("ollama", "llama3.2"),
    "skincancer-llama": ("ollama", "skincancer-llama"),
    "medgemma-skincancer": ("lmstudio", "medgemma-skincancer"),
}

DEFAULT_MODEL = "skincancer-llama"

_lmstudio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

SYSTEM_PROMPT = (
    "You are a helpful medical assistant explaining skin cancer information "
    "clearly and concisely to students and medical professionals. Never mention "
    "databases, ontologies, SPARQL or technical terms."
)

def _build_user_prompt(question, results):
    return f"""
A user asked: "{question}"

The medical knowledge base returned: {results}

Explain the answer in 2-3 clear sentences.
- Use simple medical language
- Use the exact names from the results
- Do NOT mention ontologies, SPARQL, databases or technical terms
- Just answer naturally as a medical assistant would
"""

def explain_answer(question, results, model_key=DEFAULT_MODEL):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Options: {list(MODEL_REGISTRY)}")

    backend, model_name = MODEL_REGISTRY[model_key]
    user_prompt = _build_user_prompt(question, results)

    if backend == "ollama":
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()

    elif backend == "lmstudio":
        response = _lmstudio_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{model_key}'")


if __name__ == "__main__":
    tests = [
        {
            "question": "What are the symptoms of melanoma?",
            "results": "Asymmetry\nBleeding\nColorVariation\nDiameterChange\nIrregularBorder",
        },
    ]
    for model_key in MODEL_REGISTRY:
        print(f"\n=== Testing model: {model_key} ===")
        for t in tests:
            print(explain_answer(t["question"], t["results"], model_key=model_key))