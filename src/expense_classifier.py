"""Expense categorisation: TF-IDF + Logistic Regression (scikit-learn Pipeline),
plus a keyword-rule baseline used only for comparison.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import CLASSIFIER_PATH, CLASSIFIER_REVIEW_THRESHOLD, RANDOM_SEED
from src.preprocessing import clean_description


def build_pipeline() -> Pipeline:
    """Word n-grams capture context words ('ride', 'pharmacy'); character
    n-grams help with unseen merchant spellings ('PHARMEASY' vs 'pharmacy')."""
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=clean_description, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(preprocessor=clean_description, analyzer="char_wb",
                                  ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    return Pipeline([("tfidf", features), ("clf", clf)])


class ExpenseClassifier:
    """Wraps the trained pipeline and adds confidence / review logic."""

    def __init__(self, pipeline: Pipeline | None = None, review_threshold: float = CLASSIFIER_REVIEW_THRESHOLD):
        self.pipeline = pipeline or build_pipeline()
        self.review_threshold = review_threshold

    def fit(self, descriptions, categories):
        self.pipeline.fit(list(descriptions), list(categories))
        return self

    @property
    def classes_(self):
        return list(self.pipeline.classes_)

    def predict(self, descriptions) -> pd.DataFrame:
        """Return predicted_category, confidence (max class probability), needs_review."""
        descriptions = list(descriptions)
        if not descriptions:
            return pd.DataFrame(columns=["predicted_category", "confidence", "needs_review"])
        proba = self.pipeline.predict_proba(descriptions)
        best = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        return pd.DataFrame({
            "predicted_category": np.array(self.pipeline.classes_)[best],
            "confidence": conf.round(4),
            "needs_review": conf < self.review_threshold,
        })

    def save(self, path: str | Path = CLASSIFIER_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "review_threshold": self.review_threshold}, path)

    @classmethod
    def load(cls, path: str | Path = CLASSIFIER_PATH) -> "ExpenseClassifier":
        bundle = joblib.load(path)
        return cls(bundle["pipeline"], bundle["review_threshold"])


# ---------------------------------------------------------------- baseline
class KeywordBaselineClassifier:
    """Simple hand-written keyword rules: the 'traditional' approach.

    A transaction gets the category with the most keyword hits; no hits -> 'Other'.
    Used ONLY as an evaluation baseline - the app uses the ML model.
    """
    KEYWORDS = {
        "Food": ["swiggy", "zomato", "food", "restaurant", "cafe", "pizza", "dominos", "kfc",
                 "mcdonald", "burger", "grocer", "bigbasket", "blinkit", "biryani", "meal", "dine"],
        "Transport": ["uber", "ola", "rapido", "metro", "fuel", "petrol", "irctc", "cab", "ride",
                      "taxi", "bus", "flight", "travel", "trip", "ticket"],
        "Shopping": ["amazon", "flipkart", "myntra", "ajio", "shopping", "store", "purchase",
                     "mall", "apparel", "electronics", "nykaa", "croma"],
        "Bills": ["electricity", "bill", "recharge", "postpaid", "broadband", "airtel", "jio",
                  "gas", "water", "utility", "dth", "fibernet"],
        "Entertainment": ["netflix", "spotify", "hotstar", "movie", "bookmyshow", "pvr", "inox",
                          "prime video", "game", "subscription", "streaming"],
        "Education": ["course", "udemy", "coursera", "tuition", "college", "university", "exam",
                      "book", "school", "fee"],
        "Healthcare": ["pharmacy", "medic", "hospital", "clinic", "doctor", "apollo", "health",
                       "lab", "consultation"],
        "Rent": ["rent", "landlord", "pg ", "hostel", "maintenance"],
    }

    def predict(self, descriptions) -> list[str]:
        out = []
        for text in descriptions:
            t = " " + clean_description(text) + " "
            scores = {cat: sum(k in t for k in kws) for cat, kws in self.KEYWORDS.items()}
            best = max(scores, key=scores.get)
            out.append(best if scores[best] > 0 else "Other")
        return out


# ---------------------------------------------------------------- training entry
def train_classifier(train_df: pd.DataFrame) -> ExpenseClassifier:
    """train_df needs 'description' and 'category' columns (expenses only)."""
    model = ExpenseClassifier()
    model.fit(train_df["description"], train_df["category"])
    return model
