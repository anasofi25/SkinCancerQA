import json
from collections import Counter


def format_for_finetuning(input_file, output_file):
    """Converts the generated QA dataset into Alpaca instruction-following
    format required by SFTTrainer for fine-tuning."""
    with open(input_file, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} entries")

    formatted = []
    for entry in dataset:
        formatted.append({
            "instruction": "You are a medical assistant specializing in skin cancer. Answer the following clinical question clearly and concisely.",
            "input": entry["question"],
            "output": entry["answer"]
        })

    # JSON — used for inspection
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    # JSONL — format required by SFTTrainer
    jsonl_file = output_file.replace(".json", ".jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in formatted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Plain text — human-readable version for review
    txt_file = output_file.replace(".json", ".txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("FINE-TUNING TRAINING DATA\n")
        f.write("Format: Alpaca instruction-following\n")
        f.write(f"Total entries: {len(formatted)}\n")
        f.write("=" * 60 + "\n\n")
        for i, entry in enumerate(formatted):
            f.write(f"Entry {i+1}\n")
            f.write(f"INSTRUCTION: {entry['instruction']}\n")
            f.write(f"INPUT: {entry['input']}\n")
            f.write(f"OUTPUT: {entry['output']}\n")
            f.write("-" * 60 + "\n\n")

    print(f"Saved {len(formatted)} entries to {output_file} (.json / .jsonl / .txt)")

    diag = Counter(entry["diagnostic"] for entry in dataset)
    for k, v in diag.items():
        print(f"  {k}: {v} entries")


if __name__ == "__main__":
    format_for_finetuning(
        input_file="skin_cancer_dataset_gemini40.json",
        output_file="training_data_gemini.json"
    )