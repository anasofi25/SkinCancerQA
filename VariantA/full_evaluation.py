"""
full_evaluation.py

Computes all evaluation metrics from testing_pipeline2.csv and generates
publication-ready plots for the thesis.

Metrics computed:
  1. Valid rate per model and per question category (basic / clinical / edge)
  2. Keyword hit rate per model
  3. Average retries per model
  4. Average agent3 time per model (inference speed)
  5. Cosine similarity between predicted entity and expected keyword
     (using sentence-transformers embeddings — word-level comparison
      as suggested by supervisor)
  6. Agent 2 confusion matrix (TP/TN/FP/FN) with precision/recall/F1

Install dependencies first:
    pip install pandas matplotlib scikit-learn sentence-transformers

Usage:
    python full_evaluation.py
"""

import pandas as pd
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path("../Results/testing_pipeline2.csv")
PLOTS_DIR = Path("evaluation_plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── Question category mapping ─────────────────────────────────────────────────
BASIC_QUESTIONS = {
    "What are the symptoms of melanoma?",
    "What are the symptoms of basal cell carcinoma?",
    "How is melanoma treated?",
    "How is basal cell carcinoma treated?",
    "How is squamous cell carcinoma treated?",
    "What are the risk factors for melanoma?",
    "What are the risk factors for skin cancer?",
    "What types of skin cancer exist?",
    "How is melanoma diagnosed?",
    "How is skin cancer diagnosed?",
    "Where does melanoma commonly appear?",
    "Where does basal cell carcinoma commonly appear?",
    "What symptoms does squamous cell carcinoma cause?",
    "What are the treatments for actinic keratosis?",
    "What are the risk factors for squamous cell carcinoma?",
}

EDGE_QUESTIONS = {
    "A 67-year-old male smoker has a lesion on his ear. What skin cancers commonly appear on the ear?",
    "A 68-year-old patient has a lesion on the forearm. What skin cancers are commonly found on the forearm?",
    "A 23-year-old patient has a pigmented lesion on the face that has not changed. Could this be a nevus and is it dangerous?",
    "A 73-year-old woman has basal cell carcinoma on her face. Is smoking a risk factor for this condition?",
    "A patient has a lesion that bleeds. Which skin cancers have bleeding as a symptom?",
    "What is skin cancer and how dangerous is it?",
    "A 78-year-old man who smokes and has had skin cancer before develops a lesion on his back. Which cancer has both smoking and previous skin cancer as risk factors?",
    "A 55-year-old woman has an itchy, bleeding, raised lesion on her neck. What skin cancer appears on the neck with these symptoms?",
    "A doctor is examining a suspicious mole using dermoscopy. What cancer is typically diagnosed this way?",
    "A 62-year-old female smoker has a lesion on her chest. Which skin cancers are associated with smoking?",
    "A 77-year-old patient has a persistent rough patch on the face that won't go away. Without knowing the diagnosis, what treatments exist for precancerous skin lesions?",
    "A 65-year-old patient has a rough, dark lesion on the face that looks like it could be cancerous. How would a doctor confirm whether this is malignant?",
    "A 49-year-old woman notices a mole on her thigh that has been growing, changing and becoming raised over several weeks. What is the most likely diagnosis and how is it confirmed?",
    "What is the survival rate for melanoma?",
    "A surgeon is planning Mohs surgery for a patient. Which skin cancer is most commonly treated this way?",
}


def get_category(question):
    if question in BASIC_QUESTIONS:
        return "basic"
    elif question in EDGE_QUESTIONS:
        return "edge"
    else:
        return "clinical"


# ── Entity extraction ─────────────────────────────────────────────────────────
ENTITY_MAP = {
    "melanoma": "Melanoma",
    "basal cell carcinoma": "BasalCellCarcinoma",
    "bcc": "BasalCellCarcinoma",
    "squamous cell carcinoma": "SquamousCellCarcinoma",
    "scc": "SquamousCellCarcinoma",
    "actinic keratosis": "ActinicKeratosis",
    "mohs surgery": "MohsSurgery",
    "mohs": "MohsSurgery",
    "surgical excision": "SurgicalExcision",
    "cryotherapy": "Cryotherapy",
    "immunotherapy": "Immunotherapy",
    "radiation therapy": "RadiationTherapy",
    "chemotherapy": "Chemotherapy",
    "topical therapy": "TopicalTherapy",
    "biopsy": "Biopsy",
    "dermoscopy": "Dermoscopy",
    "visual examination": "VisualExamination",
    "uv exposure": "UVExposure",
    "uvexposure": "UVExposure",
    "fair skin": "FairSkin",
    "smoking": "Smoking",
    "family history": "FamilyHistory",
    "previous skin cancer": "PreviousSkinCancer",
    "asymmetry": "Asymmetry",
    "bleeding": "Bleeding",
    "itching": "Itching",
    "elevation": "Elevation",
    "pearl": "PearlAppearance",
    "scaly": "ScalyPatch",
    "open sore": "OpenSore",
    "color variation": "ColorVariation",
    "diameter": "DiameterChange",
    "irregular border": "IrregularBorder",
    "face": "Face",
    "neck": "Neck",
    "back": "Back",
    "arms": "Arms",
    "legs": "Legs",
    "scalp": "Scalp",
}


def extract_entity(text: str) -> str:
    """Extract the most prominent medical entity from a free-text answer."""
    text_lower = text.lower()
    # sort by length descending so longer matches win (e.g. "basal cell carcinoma" before "carcinoma")
    for phrase, entity in sorted(ENTITY_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in text_lower:
            return entity
    return ""


# ── Cosine similarity ─────────────────────────────────────────────────────────
def compute_cosine_similarities(df):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    except ImportError:
        print("sentence-transformers not installed — skipping cosine similarity")
        print("Run: pip install sentence-transformers")
        df["predicted_entity"] = ""
        df["cosine_similarity"] = None
        return df

    print("\nLoading sentence-transformers model (downloads on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    predicted_entities = []
    cosine_scores = []

    for _, row in df.iterrows():
        answer = str(row.get("answer", ""))
        expected = str(row.get("expected_keyword", ""))

        if answer.startswith("FAILED") or answer.startswith("ERROR") or not answer:
            predicted_entities.append("")
            cosine_scores.append(None)
            continue

        predicted = extract_entity(answer)
        predicted_entities.append(predicted)

        if not predicted or not expected:
            cosine_scores.append(None)
            continue

        # Convert CamelCase to readable words for better embeddings
        # e.g. BasalCellCarcinoma -> "Basal Cell Carcinoma"
        def camel_to_words(s):
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

        emb_expected = model.encode([camel_to_words(expected)])
        emb_predicted = model.encode([camel_to_words(predicted)])
        score = cos_sim(emb_expected, emb_predicted)[0][0]
        cosine_scores.append(float(score))

    df["predicted_entity"] = predicted_entities
    df["cosine_similarity"] = cosine_scores
    return df


# ── Agent 2 confusion matrix ──────────────────────────────────────────────────
def compute_ground_truth(row):
    expected = str(row["expected_keyword"]).strip().lower()
    raw = str(row["raw_results"]).strip().lower()
    if not expected or raw in ("", "no results found", "nan"):
        return False
    if raw.startswith("error"):
        return False
    return expected in raw


def classify_agent2(row):
    gt = compute_ground_truth(row)
    pred = str(row["valid"]).strip().lower() == "true"
    if gt and pred:
        return "TP"
    if not gt and not pred:
        return "TN"
    if not gt and pred:
        return "FP"
    return "FN"


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_valid_rate_per_model(df):
    import matplotlib.pyplot as plt

    models = df["model"].unique()
    valid_rates = []
    for m in models:
        sub = df[df["model"] == m]
        rate = (sub["valid"].astype(str).str.lower() == "true").mean() * 100
        valid_rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, valid_rates, color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"][:len(models)])
    ax.set_ylabel("Valid rate (%)")
    ax.set_title("Pipeline Success Rate per Model")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, valid_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "valid_rate_per_model.png", dpi=200)
    plt.close()
    print("Saved: valid_rate_per_model.png")


def plot_valid_rate_per_category(df):
    import matplotlib.pyplot as plt

    categories = ["basic", "clinical", "edge"]
    models = df["model"].unique()
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        rates = []
        for cat in categories:
            sub = df[(df["model"] == model) & (df["category"] == cat)]
            if len(sub) == 0:
                rates.append(0)
            else:
                rates.append((sub["valid"].astype(str).str.lower() == "true").mean() * 100)
        ax.bar(x + i * width, rates, width=width, label=model, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["Basic (15)", "Clinical (10)", "Edge (15)"])
    ax.set_ylabel("Valid rate (%)")
    ax.set_title("Pipeline Success Rate per Question Category and Model")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "valid_rate_per_category.png", dpi=200)
    plt.close()
    print("Saved: valid_rate_per_category.png")


def plot_keyword_hit_rate(df):
    import matplotlib.pyplot as plt

    models = df["model"].unique()
    rates = []
    for m in models:
        sub = df[df["model"] == m]
        rate = (sub["keyword_hit"].astype(str).str.lower() == "true").mean() * 100
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, rates, color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"][:len(models)])
    ax.set_ylabel("Keyword hit rate (%)")
    ax.set_title("Keyword Hit Rate per Model (Faithfulness Proxy)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "keyword_hit_rate.png", dpi=200)
    plt.close()
    print("Saved: keyword_hit_rate.png")


def plot_avg_times(df):
    import matplotlib.pyplot as plt

    valid_df = df[df["valid"].astype(str).str.lower() == "true"].copy()
    valid_df["agent3_time"] = pd.to_numeric(valid_df["agent3_time"], errors="coerce")
    valid_df["total_time"] = pd.to_numeric(valid_df["total_time"], errors="coerce")

    models = valid_df["model"].unique()
    a3_times = [valid_df[valid_df["model"] == m]["agent3_time"].mean() for m in models]
    tot_times = [valid_df[valid_df["model"] == m]["total_time"].mean() for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, a3_times, width, label="Agent 3 time (s)", color="#2196F3")
    ax.bar(x + width/2, tot_times, width, label="Total pipeline time (s)", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Inference Time per Model (successful runs only)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_times.png", dpi=200)
    plt.close()
    print("Saved: avg_times.png")


def plot_cosine_similarity(df):
    import matplotlib.pyplot as plt

    cos_df = df[df["cosine_similarity"].notna()].copy()
    if cos_df.empty:
        print("No cosine similarity data to plot.")
        return

    models = cos_df["model"].unique()
    means = [cos_df[cos_df["model"] == m]["cosine_similarity"].mean() for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, means, color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"][:len(models)])
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Cosine Similarity: Predicted Entity vs Expected Entity\n(sentence-transformers embeddings)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cosine_similarity.png", dpi=200)
    plt.close()
    print("Saved: cosine_similarity.png")


def plot_confusion_matrix_overall(df):
    import matplotlib.pyplot as plt

    counts = df["agent2_class"].value_counts()
    TP = counts.get("TP", 0)
    TN = counts.get("TN", 0)
    FP = counts.get("FP", 0)
    FN = counts.get("FN", 0)

    matrix = np.array([[TP, FN], [FP, TN]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            text_color = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{value}", ha="center", va="center",
                    fontsize=14, color=text_color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted VALID", "Predicted INVALID"])
    ax.set_yticklabels(["Actually correct", "Actually wrong"])
    ax.set_title("Agent 2 Confusion Matrix\n(all models combined)")
    fig.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
    print("Saved: confusion_matrix.png")

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"\nAgent 2 classifier metrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")


def plot_retries(df):
    import matplotlib.pyplot as plt

    df["retries"] = pd.to_numeric(df["retries"], errors="coerce")
    models = df["model"].unique()
    avg_retries = [df[df["model"] == m]["retries"].mean() for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, avg_retries,
                  color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"][:len(models)])
    ax.set_ylabel("Average retries")
    ax.set_title("Average Number of SPARQL Retries per Model\n(measures Agent 1 stability)")
    ax.set_ylim(0, max(avg_retries) * 1.3 if avg_retries else 1)
    for bar, val in zip(bars, avg_retries):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_retries.png", dpi=200)
    plt.close()
    print("Saved: avg_retries.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['question'].nunique()} unique questions")

    # Add category column
    df["category"] = df["question"].apply(get_category)
    print(f"\nQuestion categories:")
    print(df.groupby("category")["question"].nunique())

    # Cosine similarity
    print("\nComputing cosine similarities...")
    df = compute_cosine_similarities(df)

    # Agent 2 classification
    df["agent2_class"] = df.apply(classify_agent2, axis=1)

    # Save enriched CSV
    enriched_path = Path("../Results/testing_pipeline2_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\nEnriched CSV saved to: {enriched_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY PER MODEL")
    print("="*70)
    models = df["model"].unique()
    print(f"{'Model':<25} {'Valid%':>7} {'KwHit%':>8} {'CosSim':>8} "
          f"{'AvgA3s':>7} {'AvgTots':>8} {'AvgRetries':>11}")
    print("-"*80)
    for m in models:
        sub = df[df["model"] == m]
        valid_pct = (sub["valid"].astype(str).str.lower() == "true").mean() * 100
        kw_pct = (sub["keyword_hit"].astype(str).str.lower() == "true").mean() * 100
        cos = sub["cosine_similarity"].mean()
        cos_str = f"{cos:.3f}" if pd.notna(cos) else "-"
        a3 = pd.to_numeric(sub["agent3_time"], errors="coerce").mean()
        tot = pd.to_numeric(sub["total_time"], errors="coerce").mean()
        ret = pd.to_numeric(sub["retries"], errors="coerce").mean()
        print(f"{m:<25} {valid_pct:>7.1f} {kw_pct:>8.1f} {cos_str:>8} "
              f"{a3:>7.1f} {tot:>8.1f} {ret:>11.2f}")

    # Generate all plots
    print("\nGenerating plots...")
    plot_valid_rate_per_model(df)
    plot_valid_rate_per_category(df)
    plot_keyword_hit_rate(df)
    plot_avg_times(df)
    plot_cosine_similarity(df)
    plot_confusion_matrix_overall(df)
    plot_retries(df)

    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}/")
    print("Done.")


if __name__ == "__main__":
    main()