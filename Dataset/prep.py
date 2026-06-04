import json

def format_for_finetuning(input_file, output_file):
    with open(input_file, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} entries")

    formatted = []

    for entry in dataset:
        # Alpaca format
        alpaca_entry = {
            "instruction": "You are a medical assistant specializing in skin cancer. Answer the following clinical question clearly and concisely.",
            "input": entry["question"],
            "output": entry["answer"]
        }
        formatted.append(alpaca_entry)

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    # Save as JSONL
    jsonl_file = output_file.replace(".json", ".jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in formatted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save as plain text
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

    print(f"Saved {len(formatted)} training entries")
    print(f"Files created:")
    print(f"  {output_file} — JSON format")
    print(f"  {jsonl_file} — JSONL format")
    print(f"  {txt_file} — readable text")

    print("\nSample entry:")
    print(f"INSTRUCTION: {formatted[0]['instruction']}")
    print(f"INPUT: {formatted[0]['input']}")
    print(f"OUTPUT: {formatted[0]['output']}")

    print("\nDiagnostic distribution in training data:")
    from collections import Counter
    diag = Counter(entry["diagnostic"] for entry in dataset)
    for k, v in diag.items():
        print(f"  {k}: {v} entries")


if __name__ == "__main__":
    format_for_finetuning(
        input_file="skin_cancer_dataset_gemini40.json",
        output_file="training_data_gemini.json"
    )