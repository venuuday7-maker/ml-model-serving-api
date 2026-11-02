"""Train the model the API serves: TF-IDF + logistic regression (gives calibrated-ish probabilities) on 4 newsgroup topics."""
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = ["sci.med", "sci.space", "rec.autos", "comp.graphics"]


def main():
    kw = dict(categories=CATEGORIES, remove=("headers", "footers", "quotes"), random_state=42)
    train, test = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
    model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)),
                      ("clf", LogisticRegression(max_iter=1000, C=10))]).fit(train.data, train.target)
    print(f"held-out accuracy: {model.score(test.data, test.target):.3f}")
    joblib.dump({"model": model, "labels": list(train.target_names)}, "model.joblib")


if __name__ == "__main__":
    main()
