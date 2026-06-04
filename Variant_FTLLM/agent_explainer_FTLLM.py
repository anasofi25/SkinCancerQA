import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = 'microsoft/Phi-3-mini-4k-instruct'
ADAPTER_PATH = r"C:\Users\Ana\PycharmProjects\skin_cancer_qaV@\phi3-skincancer-agent3"

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print("Loading fine-tuned Agent 3 model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32  # also fixed the deprecation warning
    )

    _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _tokenizer.pad_token = _tokenizer.unk_token
    _tokenizer.pad_token_id = _tokenizer.unk_token_id

    print("Model loaded.")
    return _model, _tokenizer

def explain_answer(question, results):
    print("Step 1: loading model...")
    model, tokenizer = _load_model()
    print("Step 2: model loaded, building prompt...")

    user_input = f"""A user asked: "{question}"

The medical knowledge base returned: {results}

Explain the answer in 2-3 clear sentences.
- Use simple medical language
- Use the exact names from the results
- Do NOT mention ontologies, SPARQL, databases or technical terms
- Just answer naturally as a medical assistant would"""

    messages = [
        {"role": "system", "content": "You are a medical assistant specializing in skin cancer. Answer the following clinical question clearly and concisely."},
        {"role": "user", "content": user_input}
    ]

    print("Step 3: applying chat template...")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("Step 4: tokenizing...")
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    print("Step 5: generating...")
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            eos_token_id=tokenizer.eos_token_id
        )

    print("Step 6: decoding...")
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "<|assistant|>" in full_response:
        answer = full_response.split("<|assistant|>")[-1].strip()
    else:
        answer = full_response.split(user_input)[-1].strip()

    return answer


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
        print(f"Results:  {test['results']}")
        print("\nAnswer:")
        answer = explain_answer(test["question"], test["results"])
        print(answer)

    print("\nDone")