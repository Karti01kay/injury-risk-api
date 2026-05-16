"""
Model training pipeline for Injury Risk Forecaster.
Trains a Random Forest classifier, evaluates it thoroughly,
and exports the model + scaler for API serving.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score,
)
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import CalibratedClassifierCV

from data_generator    import generate_dataset
from feature_engineering import engineer_features, get_feature_columns

import os as _os
OUT_DIR = _os.environ.get("OUTPUTS_DIR",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "api", "outputs"))
_os.makedirs(OUT_DIR, exist_ok=True)



warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)

SEED   = 42
TARGET = "injury_occurred"


# ── 1. Generate + engineer data ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Injury Risk Forecaster: ML Training Pipeline")
print("=" * 60)

print("\n[1/6] Generating synthetic athlete data …")
raw = generate_dataset(n_athletes=300, days=180)
df  = engineer_features(raw)

FEATURES = get_feature_columns()
# Keep only columns that actually exist (sport dummies depend on data)
FEATURES = [c for c in FEATURES if c in df.columns]

X = df[FEATURES]
y = df[TARGET]

print(f"       Samples  : {len(X):,}")
print(f"       Features : {len(FEATURES)}")
print(f"       Injury % : {y.mean()*100:.1f}%")


# ── 2. Train / test split ────────────────────────────────────────────────────
print("\n[2/6] Splitting data (80/20 stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)


# ── 3. Build pipelines ───────────────────────────────────────────────────────
print("\n[3/6] Training candidate models …")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

candidates = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=SEED,
    ),
    "Logistic Regression": LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=SEED,
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, clf in candidates.items():
    X_fit = X_train_s if name == "Logistic Regression" else X_train_s
    cv_scores = cross_val_score(clf, X_train_s, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    f1     = f1_score(y_test, clf.predict(X_test_s))
    results[name] = {"model": clf, "auc": auc, "ap": ap, "f1": f1,
                     "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    print(f"  {name:25s}  CV-AUC={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
          f"  Test-AUC={auc:.3f}  AP={ap:.3f}  F1={f1:.3f}")


# ── 4. Select best model ─────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
clf       = best["model"]
print(f"\n  → Best model: {best_name}  (AUC={best['auc']:.3f})")

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:, 1]

print("\n[4/6] Classification report:")
print(classification_report(y_test, y_pred, target_names=["No Injury", "Injury"]))


# ── 5. Visualisations ────────────────────────────────────────────────────────
print("[5/6] Generating evaluation plots …")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Injury Risk Forecaster — Model Evaluation", fontsize=14, fontweight="bold")

# (a) ROC curve
ax = axes[0, 0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["model"].predict_proba(X_test_s)[:, 1])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (b) Precision-Recall curve
ax = axes[0, 1]
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(
        y_test, r["model"].predict_proba(X_test_s)[:, 1])
    ax.plot(rec, prec, label=f"{name} (AP={r['ap']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (c) Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Injury", "Injury"],
            yticklabels=["No Injury", "Injury"])
ax.set_title(f"Confusion Matrix — {best_name}")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# (d) Feature importance (top 15)
ax = axes[1, 1]
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    top = imp.head(15)
    colors = ["#378ADD" if "acwr" in i or "load" in i
              else "#1D9E75" if "sleep" in i
              else "#BA7517" if "soren" in i or "rhr" in i
              else "#888780" for i in top.index]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR + "/model_evaluation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("       Saved → outputs/model_evaluation.png")


# ── 6. Save artefacts ────────────────────────────────────────────────────────
print("\n[6/6] Saving model artefacts …")

joblib.dump(clf,    OUT_DIR + "/injury_model.joblib")
joblib.dump(scaler, OUT_DIR + "/scaler.joblib")

meta = {
    "model_type":   best_name,
    "features":     FEATURES,
    "n_features":   len(FEATURES),
    "test_auc":     round(best["auc"], 4),
    "test_ap":      round(best["ap"],  4),
    "test_f1":      round(best["f1"],  4),
    "cv_auc_mean":  round(best["cv_mean"], 4),
    "cv_auc_std":   round(best["cv_std"],  4),
    "risk_thresholds": {
        "low":      0.25,
        "moderate": 0.50,
        "high":     0.70,
    },
    "trained_on": str(pd.Timestamp.now().date()),
}

with open(OUT_DIR + "/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"       injury_model.joblib  ✓")
print(f"       scaler.joblib        ✓")
print(f"       model_meta.json      ✓")

print("\n" + "=" * 60)
print("Phase 1 complete!")
print(f"  Best model : {best_name}")
print(f"  Test AUC   : {best['auc']:.4f}")
print(f"  Test AP    : {best['ap']:.4f}")
print(f"  Test F1    : {best['f1']:.4f}")
print("=" * 60)