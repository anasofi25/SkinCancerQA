from pipeline_FTLLM import run_pipeline_with_timeout
import datetime

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

quick_test_questions = [
    {"question": "What are the symptoms of melanoma?", "expected": "Asymmetry"},
    {"question": "How is squamous cell carcinoma treated?", "expected": "SurgicalExcision"},
    {"question": "What are the risk factors for melanoma?", "expected": "UVExposure"},
    {"question": "What symptoms does squamous cell carcinoma cause?", "expected": "ScalyPatch"},
    {"question": "What are the treatments for actinic keratosis?", "expected": "Cryotherapy"},
]

medqa_questions = [
    {"question": "What type of melanoma is most common in people with darker skin?", "expected": "Acral"},
    {"question": "What is the most likely diagnosis for a dome-shaped plaque with central keratin plug from sun exposure?", "expected": "Keratoacanthoma"},
    {"question": "What is the most likely diagnosis for a lesion on the scalp of a retired gardener?", "expected": "Actinic"},
    {"question": "What color characteristics would be found in a melanocytic lesion?", "expected": "color"},
    {"question": "What is the treatment for metastatic melanoma with aldesleukin?", "expected": "immune"},
    {"question": "Which cancer type has the highest tendency to cause brain metastasis?", "expected": "Melanoma"},
    {"question": "What is the most likely diagnosis for a painless ulcer on the lower lip with sun exposure?", "expected": "Squamous"},
    {"question": "What is the diagnosis for brown greasy lesions on the forehead that cannot be peeled off?", "expected": "Seborrheic"},
    {"question": "What is the diagnosis for a hyperpigmented papule that retracts inward when squeezed?", "expected": "Dermatofibroma"},
]


def check_answer(answer: str, expected: str) -> bool:
    """Normalize camelCase and spaces before comparing."""
    # Remove spaces and hyphens, lowercase both
    answer_norm = answer.lower().replace(" ", "").replace("-", "")
    expected_norm = expected.lower().replace(" ", "").replace("-", "")
    return expected_norm in answer_norm


def run_test_suite(name: str, questions: list, log_file=None):
    print("\n" + "=" * 60)
    print(f"TEST SUITE: {name}")
    print("=" * 60)

    passed = 0
    failed = 0
    failed_questions = []
    log_lines = []

    log_lines.append("=" * 60)
    log_lines.append(f"TEST SUITE: {name}")
    log_lines.append("=" * 60)

    for item in questions:
        question = item["question"]
        expected = item["expected"]

        print(f"\n {question}")
        print(f"   Expected keyword: '{expected}'")

        sparql, results, answer = run_pipeline_with_timeout(question)

        if answer is None:
            print(f"    FAILED — pipeline returned no answer")
            failed += 1
            failed_questions.append({"question": question, "expected": expected, "answer": "NO ANSWER"})
            log_lines.append(f"\nQ: {question}")
            log_lines.append(f"Expected: {expected}")
            log_lines.append(f"Answer: NO ANSWER")
            log_lines.append(f"Result:  FAIL")
            continue

        print(f"    Answer: {answer}")

        if check_answer(answer, expected):
            print(f"  PASS — '{expected}' found in answer")
            passed += 1
            status = " PASS"
        else:
            print(f"    FAIL — '{expected}' NOT found in answer")
            failed += 1
            failed_questions.append({"question": question, "expected": expected, "answer": answer})
            status = " FAIL"

        log_lines.append(f"\nQ: {question}")
        log_lines.append(f"Expected keyword: {expected}")
        log_lines.append(f"Answer: {answer}")
        log_lines.append(f"Result: {status}")
        log_lines.append("-" * 40)

    # Summary
    total = passed + failed
    summary = f"RESULTS: {passed}/{total} passed  ({round(passed/total*100)}% accuracy)"
    print("\n" + "-" * 60)
    print(summary)

    log_lines.append(f"\n{summary}")

    if failed_questions:
        print("\nFailed questions:")
        log_lines.append("\nFailed questions:")
        for f in failed_questions:
            print(f"  • {f['question']}")
            print(f"    Expected: {f['expected']}")
            print(f"    Got: {f['answer'][:120]}...")
            log_lines.append(f"  • {f['question']}")
            log_lines.append(f"    Expected: {f['expected']}")
            log_lines.append(f"    Got: {f['answer'][:120]}")

    if log_file:
        log_file.write("\n".join(log_lines) + "\n\n")
        log_file.flush()

    return passed, failed

if __name__ == "__main__":
    SUITE_TO_RUN = "medqa"  #change

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("evaluation_FTLLM_2.txt", "w", encoding="utf-8") as log_file:
        log_file.write(f"EVALUATION REPORT — Fine-Tuned LLM (Agent 3)\n")
        log_file.write(f"Date: {timestamp}\n")
        log_file.write(f"Model: agent3-skin-cancer (Llama 3.2 3B, LoRA fine-tuned)\n")
        log_file.write("=" * 60 + "\n\n")

        if SUITE_TO_RUN == "quick":
            run_test_suite("Quick Test (5 questions)", quick_test_questions, log_file)

        elif SUITE_TO_RUN == "custom":
            run_test_suite("Custom Questions (15 questions)", custom_questions, log_file)

        elif SUITE_TO_RUN == "medqa":
            run_test_suite("MedQA Questions (9 questions)", medqa_questions, log_file)

        elif SUITE_TO_RUN == "all":
            p1, f1 = run_test_suite("Quick Test", quick_test_questions, log_file)
            p2, f2 = run_test_suite("Custom Questions", custom_questions, log_file)
            p3, f3 = run_test_suite("MedQA Questions", medqa_questions, log_file)

            total_p = p1 + p2 + p3
            total_f = f1 + f2 + f3
            total = total_p + total_f
            overall = f"OVERALL: {total_p}/{total} passed ({round(total_p/total*100)}% accuracy)"

            print("\n" + "=" * 60)
            print(overall)

            log_file.write("=" * 60 + "\n")
            log_file.write(overall + "\n")

        print("\n Results saved to evaluation_FTLLM.txt")