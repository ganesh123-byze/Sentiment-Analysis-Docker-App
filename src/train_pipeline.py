import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from src.data_preprocessing import TextPreprocessor


# -----------------------------
# 1️⃣ Load Dataset (Sentiment140)
# -----------------------------
print("Loading dataset...")

data = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

data.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Keep only required columns
data = data[["sentiment", "text"]]

# Convert labels: 0 → negative, 4 → positive
data["sentiment"] = data["sentiment"].replace({0: "negative", 4: "positive"})

# Use larger sample for better learning
data = data.sample(100000, random_state=42)

print("Dataset loaded successfully!")
print("Total samples used:", len(data))


# -----------------------------
# 2️⃣ Preprocessing
# -----------------------------
print("Preprocessing text...")

processor = TextPreprocessor()
data["clean_text"] = data["text"].apply(processor.clean_text)

X = data["clean_text"]
y = data["sentiment"]


# -----------------------------
# 3️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 4️⃣ Improved ML Pipeline
# -----------------------------
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,      # Larger vocabulary
            ngram_range=(1, 2),      # Unigrams + Bigrams
            min_df=5,                # Ignore rare words
            max_df=0.9,              # Ignore very common words
            sublinear_tf=True        # Better scaling
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=500,
            C=2,
            solver="lbfgs"
        )
    )
])


# -----------------------------
# 5️⃣ Train Model
# -----------------------------
print("Training model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")


# -----------------------------
# 6️⃣ Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# 7️⃣ Save Model
# -----------------------------
joblib.dump(pipeline, "models/sentiment_model.pkl")

print("\nModel saved successfully in models/sentiment_model.pkl")
