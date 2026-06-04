import csv
import json
import time
import re
from google import genai
from google.genai import types

# Configure Gemini
API_KEY = ""
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"
#pprMODEL = "gemini-2.0-flash"

DIAGNOSIS_MAP = {
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "SCC": "Squamous Cell Carcinoma",
    "ACK": "Actinic Keratosis",
    "SEK": "Seborrheic Keratosis",
    "NEV": "Nevus (Mole)"
}

def build_clinical_description(row):
    age = row["age"] or "unknown age"
    gender = row["gender"].lower() if row["gender"] else "patient"
    region = row["region"].lower() if row["region"] else "unknown region"
    diagnosis = DIAGNOSIS_MAP.get(row["diagnostic"], row["diagnostic"])

    symptoms = []
    if row["itch"] == "True": symptoms.append("itching")
    if row["grew"] == "True": symptoms.append("growth over time")
    if row["hurt"] == "True": symptoms.append("pain")
    if row["changed"] == "True": symptoms.append("changes in appearance")
    if row["bleed"] == "True": symptoms.append("bleeding")
    if row["elevation"] == "True": symptoms.append("elevation")

    risk_factors = []
    if row["smoke"] == "True": risk_factors.append("smoking")
    if row["pesticide"] == "True": risk_factors.append("pesticide exposure")
    if row["skin_cancer_history"] == "True": risk_factors.append("previous skin cancer")
    if row["cancer_history"] == "True": risk_factors.append("cancer history")

    diameter = ""
    if row["diameter_1"] and row["diameter_2"]:
        diameter = f"Lesion diameter: {row['diameter_1']}mm x {row['diameter_2']}mm."

    skin_type = f"Fitzpatrick skin type {row['fitspatrick']}." if row["fitspatrick"] else ""

    description = f"""Patient: {age}-year-old {gender}
Location: {region}
Symptoms: {', '.join(symptoms) if symptoms else 'none reported'}
Risk factors: {', '.join(risk_factors) if risk_factors else 'none reported'}
{diameter}
{skin_type}
Biopsy performed: {row['biopsed']}
Diagnosis: {diagnosis}""".strip()

    return description, diagnosis


def generate_qa_pair(clinical_description, diagnosis, max_retries=5):
    prompt = f"""You are a medical education expert creating training data for a skin cancer QA system.

Clinical case:
{clinical_description}

Generate a question and answer about this case.

Return JSON only:
{{"question": "one clinical question", "answer": "2-3 sentence answer mentioning {diagnosis}"}}

Keep the answer under 100 words."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                )
            )

            if response is None or not hasattr(response, 'text') or response.text is None:
                raise ValueError("Empty response from Gemini - rate limited")

            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            json_match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group()

            result = json.loads(text)

            if "question" not in result or "answer" not in result:
                raise ValueError("Missing question or answer field")

            return result

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate limited" in error_str.lower() or "Empty response" in error_str:
                wait_time = 60 * (attempt + 1)
                print(f"  Rate limit (attempt {attempt+1}) — waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                print(f"  Attempt {attempt+1} failed: {e} — retrying in 6s...")
                time.sleep(6)
            else:
                raise


def generate_dataset(input_csv, output_file):

    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    usable_rows = [
        r for r in rows
        if r["diagnostic"] in DIAGNOSIS_MAP
        and r["age"]
        and r["region"]
    ]

    print(f"Total usable rows in CSV: {len(usable_rows)}")

    #Count by type
    from collections import Counter
    type_counts = Counter(r["diagnostic"] for r in usable_rows)
    for diag, count in sorted(type_counts.items()):
        print(f"  {diag} ({DIAGNOSIS_MAP[diag]}): {count} rows")

    # Load existing progress
    dataset = []
    try:
        with open(output_file, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"\nResuming from {len(dataset)} existing entries")
    except FileNotFoundError:
        print("\nStarting fresh")

    #avoid duplicates
    # Use (age, gender, region, diagnostic) as a loose fingerprint
    done_fingerprints = set()
    for entry in dataset:
        lines = entry["clinical_data"].split("\n")
        done_fingerprints.add(entry.get("row_index", -1))

    done_indices = {entry.get("row_index") for entry in dataset if "row_index" in entry}
    # Fallback
    already_done_count = len(dataset) - len(done_indices)
    fallback_skip = already_done_count

    print(f"Entries with tracked index: {len(done_indices)}")
    print(f"Entries without index (will skip first {fallback_skip} unindexed rows): {already_done_count}")
    print(f"\nProcessing {len(usable_rows)} total rows...")
    print("=" * 60)

    failed = 0
    skipped = 0
    fallback_counter = 0  # counts unindexed rows

    for i, row in enumerate(usable_rows):
        # Skip if already processed by index
        if i in done_indices:
            skipped += 1
            continue

        if fallback_counter < fallback_skip:
            fallback_counter += 1
            skipped += 1
            continue

        diagnosis_label = DIAGNOSIS_MAP.get(row["diagnostic"], row["diagnostic"])
        print(f"[{len(dataset)+1}] Row {i} — {row['diagnostic']} ({diagnosis_label})...")

        try:
            description, diagnosis = build_clinical_description(row)
            qa = generate_qa_pair(description, diagnosis)

            dataset.append({
                "id": len(dataset) + 1,
                "row_index": i,
                "diagnostic": row["diagnostic"],
                "diagnosis_full": diagnosis,
                "clinical_data": description,
                "question": qa["question"],
                "answer": qa["answer"]
            })

            # Save after every entry
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            print(f"  Q: {qa['question'][:80]}...")
            print(f"  A: {qa['answer'][:80]}...")

            time.sleep(4)

        except Exception as e:
            print(f"  FAILED on row {i}: {e}")
            failed += 1
            continue

    # Save plain text version
    txt_file = output_file.replace(".json", ".txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("SKIN CANCER QA DATASET\n")
        f.write("Generated from PAD-UFES-20\n")
        f.write(f"Total entries: {len(dataset)}\n")
        f.write("=" * 60 + "\n\n")
        for entry in dataset:
            f.write(f"ID: {entry['id']} | Type: {entry['diagnosis_full']}\n")
            f.write(f"Q: {entry['question']}\n")
            f.write(f"A: {entry['answer']}\n")
            f.write("-" * 60 + "\n\n")

    print(f"\n{'='*60}")
    print(f"Total in dataset: {len(dataset)}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Saved to: {output_file}")
    print(f"Saved to: {txt_file}")


if __name__ == "__main__":
    generate_dataset(
        input_csv="metadata.csv",
        output_file="skin_cancer_dataset_gemini40.json",  # same file — resumes automatically
    )