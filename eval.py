import datetime
from VariantA.pipeline_v2 import run_pipeline

custom_questions = [
    {"question": "What are the symptoms of melanoma?", "expected": "Asymmetry"},
    {"question": "What are the symptoms of basal cell carcinoma?", "expected": "PearlAppearance"},
    {"question": "How is melanoma treated?", "expected": "Immunotherapy"},
    {"question": "How is basal cell carcinoma treated?", "expected": "MohsSurgery"},
    {"question": "How is squamous cell carcinoma treated?", "expected": "SurgicalExcision"},
    {"question": "What are the risk factors for melanoma?", "expected": "UVExposure"},
    {"question": "What are the risk factors for skin cancer?", "expected": "UVExposure"},
    {"question": "What types of skin cancer exist?", "expected": "Melanoma"},
    {"question": "How is melanoma diagnosed?", "expected": "Biopsy"},
    {"question": "How is skin cancer diagnosed?", "expected": "Biopsy"},
    {"question": "Where does melanoma commonly appear?", "expected": "Back"},
    {"question": "Where does basal cell carcinoma commonly appear?", "expected": "Face"},
    {"question": "What symptoms does squamous cell carcinoma cause?", "expected": "ScalyPatch"},
    {"question": "What are the treatments for actinic keratosis?", "expected": "Cryotherapy"},
    {"question": "What are the risk factors for squamous cell carcinoma?", "expected": "Smoking"},
]

# run only custom questions
all_questions = custom_questions

results = []
correct = 0
total = len(all_questions)

print(f"Running evaluation on {total} questions...")
print("=" * 60)

for i, item in enumerate(all_questions):
    q = item["question"]
    expected = item["expected"].lower()

    print(f"\n[{i+1}/{total}] {q[:55]}...")

    sparql, raw_results, answer = run_pipeline(q)

    if answer:
        answer_lower = answer.lower()
        raw_lower = (raw_results or "").lower()
        is_correct = expected in answer_lower or expected in raw_lower
    else:
        is_correct = False
        answer = "FAILED - no results"
        raw_results = ""
        sparql = ""

    if is_correct:
        correct += 1
        print(f"CORRECT - expected '{expected}' found")
    else:
        print(f"INCORRECT - expected '{expected}' not found")

    results.append({
        "id": i + 1,
        "question": q,
        "expected": item["expected"],
        "sparql": sparql or "",
        "raw_results": raw_results or "",
        "answer": answer,
        "correct": is_correct
    })

# calculate scores
accuracy = (correct / total) * 100
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# save evaluation report
with open("Results/evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("EVALUATION REPORT — VARIANT A (Prompted LLM)\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Model: llama3.2 via Ollama\n")
    f.write(f"Ontology: skinCancerExpanded\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"OVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)\n\n")
    f.write("=" * 60 + "\n")
    f.write("DETAILED RESULTS\n")
    f.write("=" * 60 + "\n\n")

    for r in results:
        status = "CORRECT" if r["correct"] else "INCORRECT"
        f.write(f"Q{r['id']}: {r['question']}\n")
        f.write(f"Expected keyword: {r['expected']}\n")
        f.write(f"SPARQL: {r['sparql']}\n")
        f.write(f"Raw results: {r['raw_results']}\n")
        f.write(f"Answer: {r['answer']}\n")
        f.write(f"Result: {status}\n")
        f.write("-" * 60 + "\n\n")

# save clean question-answer pairs separately
with open("Results/qa_pairs_variant_a.txt", "w", encoding="utf-8") as f:
    f.write("QUESTION-ANSWER PAIRS — VARIANT A\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Model: llama3.2 via Ollama (prompted, no fine-tuning)\n")
    f.write("=" * 60 + "\n\n")

    for r in results:
        f.write(f"Q: {r['question']}\n")
        f.write(f"A: {r['answer']}\n\n")

print(f"\n{'='*60}")
print(f"FINAL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
print(f"\nSaved to:")
print(f"  evaluation_report.txt  — full results with SPARQL")
print(f"  qa_pairs_variant_a.txt — clean Q&A pairs")