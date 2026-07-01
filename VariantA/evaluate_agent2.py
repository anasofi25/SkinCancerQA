"""
evaluate_agent2_classifier.py

Treats Agent 2 as a binary classifier (valid / invalid) and evaluates it
against a ground-truth label derived independently from the data:

    ground_truth = True  if expected_keyword actually appears in raw_results
                          (i.e. the ontology DID return the correct answer)
    ground_truth = False if expected_keyword does NOT appear in raw_results
                          (i.e. the ontology did not return the correct answer,
                           either because of an empty result set, an error,
                           or a genuinely wrong/irrelevant result)

    prediction    = Agent 2's own "valid" decision

This produces a standard confusion matrix:

    TP: Agent 2 said valid, and the correct answer was indeed present
        -> Agent 2 correctly accepted a good result
    TN: Agent 2 said invalid, and the correct answer was indeed absent
        -> Agent 2 correctly rejected a bad/empty result
    FP: Agent 2 said valid, but the correct answer was NOT present
        -> Agent 2 incorrectly accepted a wrong/irrelevant result
    FN: Agent 2 said invalid, but the correct answer WAS present
        -> Agent 2 incorrectly rejected a good result (the rule-based
           false positive case discussed for the "diagnosed as SCC"
           question is a textbook example of this)

Usage:
    python evaluate_agent2_classifier.py
"""

import pandas as pd
from pathlib import Path

CSV_PATH = Path("../Results/testing_pipeline2.csv")  # change if needed
OUTPUT_PATH = Path("../Results/agent2_classifier_results.csv")


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["expected_keyword", "raw_results", "valid"])
    return df


def compute_ground_truth(row):
    """True if the expected keyword is present in what GraphDB actually
    returned, regardless of what Agent 2 decided."""
    expected = str(row["expected_keyword"]).strip().lower()
    raw = str(row["raw_results"]).strip().lower()
    if not expected or raw in ("", "no results found", "nan"):
        return False
    if raw.startswith("error"):
        return False
    return expected in raw


def compute_prediction(row):
    val = str(row["valid"]).strip().lower()
    return val == "true"


def classify(row):
    gt = row["ground_truth"]
    pred = row["prediction"]
    if gt and pred:
        return "TP"
    if not gt and not pred:
        return "TN"
    if not gt and pred:
        return "FP"
    if gt and not pred:
        return "FN"


def main():
    df = load_data()
    df["ground_truth"] = df.apply(compute_ground_truth, axis=1)
    df["prediction"] = df.apply(compute_prediction, axis=1)
    df["classification"] = df.apply(classify, axis=1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Detailed results saved to: {OUTPUT_PATH.resolve()}\n")

    # ── Overall confusion matrix ────────────────────────────────────────────
    counts = df["classification"].value_counts()
    TP = counts.get("TP", 0)
    TN = counts.get("TN", 0)
    FP = counts.get("FP", 0)
    FN = counts.get("FN", 0)

    print("=" * 50)
    print("OVERALL CONFUSION MATRIX (Agent 2 as classifier)")
    print("=" * 50)
    print(f"                  Predicted VALID   Predicted INVALID")
    print(f"Actually correct      TP={TP:<4}           FN={FN:<4}")
    print(f"Actually wrong        FP={FP:<4}           TN={TN:<4}")
    print()

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}  (of results Agent 2 accepted, how many were actually correct)")
    print(f"Recall:    {recall:.3f}  (of results that were actually correct, how many Agent 2 accepted)")
    print(f"F1 score:  {f1:.3f}")

    # ── Per-model breakdown ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PER-MODEL BREAKDOWN")
    print("=" * 50)

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        c = sub["classification"].value_counts()
        tp, tn, fp, fn = c.get("TP", 0), c.get("TN", 0), c.get("FP", 0), c.get("FN", 0)
        tot = tp + tn + fp + fn
        acc = (tp + tn) / tot if tot else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"\n{model}")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    # ── Show the FN cases (most interesting for the thesis) ────────────────
    fn_cases = df[df["classification"] == "FN"]
    if len(fn_cases) > 0:
        print("\n" + "=" * 50)
        print(f"FALSE NEGATIVE CASES ({len(fn_cases)}) — Agent 2 wrongly rejected correct results")
        print("=" * 50)
        for _, row in fn_cases.iterrows():
            print(f"\nQ: {row['question'][:80]}...")
            print(f"  Model: {row['model']}")
            print(f"  Expected: {row['expected_keyword']}")
            print(f"  raw_results contained it but Agent 2 said: valid={row['valid']}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_confusion_matrix(TP, TN, FP, FN)
    plot_per_model_breakdown(df)


def plot_confusion_matrix(TP, TN, FP, FN):
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[TP, FN],
                        [FP, TN]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            label = labels[i][j]
            text_color = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, f"{label}\n{value}", ha="center", va="center",
                     fontsize=14, color=text_color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted VALID", "Predicted INVALID"])
    ax.set_yticklabels(["Actually correct", "Actually wrong"])
    ax.set_title("Agent 2 Confusion Matrix\n(validation outcome vs. ground truth)")
    fig.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig("evaluation_plots/agent2_confusion_matrix.png", dpi=200)
    print("\nSaved plot: agent2_confusion_matrix.png")
    plt.close()


def plot_per_model_breakdown(df):
    import matplotlib.pyplot as plt

    models = df["model"].unique()
    categories = ["TP", "TN", "FP", "FN"]
    data = {cat: [] for cat in categories}

    for model in models:
        sub = df[df["model"] == model]
        counts = sub["classification"].value_counts()
        for cat in categories:
            data[cat].append(counts.get(cat, 0))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(models))
    width = 0.2
    colors = {"TP": "#4CAF50", "TN": "#2196F3", "FP": "#FF9800", "FN": "#F44336"}

    for i, cat in enumerate(categories):
        positions = [p + i * width for p in x]
        ax.bar(positions, data[cat], width=width, label=cat, color=colors[cat])

    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Agent 2 Classification Outcomes per Model")
    ax.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig("evaluation_plots/agent2_per_model_breakdown.png", dpi=200)
    print("Saved plot: agent2_per_model_breakdown.png")
    plt.close()


if __name__ == "__main__":
    main()