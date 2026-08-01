import pandas as pd
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from google import genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

print("All libraries imported successfully!")

train = pd.read_json("Project/dataset/train.json", lines=True)
validation = pd.read_json("Project/dataset/validation.json", lines=True)
test = pd.read_json("Project/dataset/test.json", lines=True)

print("Train Records:", len(train))
print("Validation Records:", len(validation))
print("Test Records:", len(test))

print("\nColumns:")
print(train.columns)
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

train["sentiment"] = train["transcript"].apply(get_sentiment)

print(train[["transcript", "sentiment"]].head())
print(train["sentiment"].value_counts())

print("\n==============================")
print("Missing Values")
print("==============================")

print(train.isnull().sum())

print("\nValidation Missing Values")
print(validation.isnull().sum())

print("\nTest Missing Values")
print(test.isnull().sum())

print("\n==============================")
print("Duplicate Records")
print("==============================")

print("Train :", train.duplicated().sum())
print("Validation :", validation.duplicated().sum())
print("Test :", test.duplicated().sum())

import matplotlib.pyplot as plt

# Feature Engineering
train["Transcript_Length"] = train["transcript"].apply(lambda x: len(x.split()))
train["Summary_Length"] = train["summary"].apply(lambda x: len(x.split()))

print("\nTranscript Statistics")
print(train["Transcript_Length"].describe())

print("\nSummary Statistics")
print(train["Summary_Length"].describe())

plt.figure(figsize=(8,5))

plt.hist(train["Transcript_Length"], bins=30)

plt.title("Transcript Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.savefig("transcript_length_distribution.png")

plt.close()

print("Graph saved successfully.")

plt.figure(figsize=(8,5))

plt.hist(train["Summary_Length"], bins=25)

plt.title("Summary Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.savefig("summary_length_distribution.png")

plt.close()

print("Summary graph saved successfully.")

plt.figure(figsize=(6,4))

train["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("sentiment_distribution.png")

plt.close()

print("Sentiment graph saved successfully.")

plt.figure(figsize=(8,5))

plt.scatter(
    train["Transcript_Length"],
    train["Summary_Length"],
    alpha=0.5
)

plt.title("Transcript vs Summary Length")
plt.xlabel("Transcript Length")
plt.ylabel("Summary Length")

plt.savefig("transcript_summary_scatter.png")

plt.close()

print("Scatter plot saved successfully.")

# Features and Labels
X_train = train["transcript"]
y_train = train["sentiment"]

X_test = test["transcript"]

# Create sentiment labels for test data
test["sentiment"] = test["transcript"].apply(get_sentiment)
y_test = test["sentiment"]
le = LabelEncoder()

y_train_xgb = le.fit_transform(y_train)
y_test_xgb = le.transform(y_test)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed.")

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model Training Completed!")

print("\nMaking Predictions...")

y_pred = model.predict(X_test_tfidf)

print("Prediction Completed!")


print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n===================================")
print("Training Random Forest Model...")
print("===================================")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1
)

print("Running Hyperparameter Tuning...")

grid_search.fit(X_train_tfidf, y_train)

print("Hyperparameter Tuning Completed!")

print("Best Parameters:", grid_search.best_params_)

rf_model = grid_search.best_estimator_

rf_pred = rf_model.predict(X_test_tfidf)

print("Random Forest Prediction Completed!")

print("Prediction Completed!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from xgboost import XGBClassifier


print("\n========== RANDOM FOREST RESULTS ==========")

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average="weighted", zero_division=0)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, rf_pred, zero_division=0))

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix")
print(rf_cm)

print("\n===================================")
print("Training Naive Bayes Model...")
print("===================================")

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

print("Naive Bayes Training Completed!")

nb_pred = nb_model.predict(X_test_tfidf)

print("Prediction Completed!")

print("\n========== NAIVE BAYES RESULTS ==========")

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred, average="weighted", zero_division=0)
nb_recall = recall_score(y_test, nb_pred, average="weighted", zero_division=0)
nb_f1 = f1_score(y_test, nb_pred, average="weighted", zero_division=0)

print(f"Accuracy : {nb_accuracy:.4f}")
print(f"Precision : {nb_precision:.4f}")
print(f"Recall : {nb_recall:.4f}")
print(f"F1 Score : {nb_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, nb_pred, zero_division=0))

nb_cm = confusion_matrix(y_test, nb_pred)

print("\nNaive Bayes Confusion Matrix")
print(nb_cm)

print("\n===================================")
print("Training Support Vector Machine Model...")
print("===================================")

svm_model = LinearSVC(random_state=42)

svm_model.fit(X_train_tfidf, y_train)

print("SVM Training Completed!")

svm_pred = svm_model.predict(X_test_tfidf)

print("Prediction Completed!")

print("\n========== SVM RESULTS ==========")

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_recall = recall_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average="weighted", zero_division=0)

print(f"Accuracy : {svm_accuracy:.4f}")
print(f"Precision : {svm_precision:.4f}")
print(f"Recall : {svm_recall:.4f}")
print(f"F1 Score : {svm_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, svm_pred, zero_division=0))

svm_cm = confusion_matrix(y_test, svm_pred)

print("\nSVM Confusion Matrix")
print(svm_cm)

print("\n===================================")
print("Training XGBoost Model...")
print("===================================")

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train_tfidf, y_train_xgb)

print("XGBoost Training Completed!")

xgb_pred = xgb_model.predict(X_test_tfidf)


print("Prediction Completed!")

print("\n========== XGBOOST RESULTS ==========")

xgb_accuracy = accuracy_score(y_test_xgb, xgb_pred)
xgb_precision = precision_score(y_test_xgb, xgb_pred, average="weighted", zero_division=0)
xgb_recall = recall_score(y_test_xgb, xgb_pred, average="weighted", zero_division=0)
xgb_f1 = f1_score(y_test_xgb, xgb_pred, average="weighted", zero_division=0)

print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision : {xgb_precision:.4f}")
print(f"Recall : {xgb_recall:.4f}")
print(f"F1 Score : {xgb_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test_xgb, xgb_pred, zero_division=0))

xgb_cm = confusion_matrix(y_test_xgb, xgb_pred)

print("\nXGBoost Confusion Matrix")
print(xgb_cm)

print("\n================ MODEL COMPARISON ================")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Naive Bayes",
        "Support Vector Machine",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        nb_accuracy,
        svm_accuracy,
        xgb_accuracy
    ],
    "Precision": [
        precision,
        rf_precision,
        nb_precision,
        svm_precision,
        xgb_precision
    ],
    "Recall": [
        recall,
        rf_recall,
        nb_recall,
        svm_recall,
        xgb_recall
    ],
    "F1 Score": [
        f1,
        rf_f1,
        nb_f1,
        svm_f1,
        xgb_f1
    ]
})

print(comparison)

comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")


sample_train = train.head(5).copy()
sample_test = test.head(2).copy()

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values
#-----------------------------rest of the google embedding code------------

print("\nGenerating Gemini Embeddings...")

X_train_embed = sample_train["transcript"].apply(get_embedding).tolist()
X_test_embed = sample_test["transcript"].apply(get_embedding).tolist()

print("Embeddings Generated Successfully!")

y_train_embed = sample_train["sentiment"]
y_test_embed = sample_test["sentiment"]

embed_model = LogisticRegression(max_iter=1000)

embed_model.fit(X_train_embed, y_train_embed)

embed_pred = embed_model.predict(X_test_embed)

print("\n========== GEMINI EMBEDDING MODEL ==========")

print("Accuracy :", accuracy_score(y_test_embed, embed_pred))
print("Precision :", precision_score(y_test_embed, embed_pred, average="weighted", zero_division=0))
print("Recall :", recall_score(y_test_embed, embed_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_test_embed, embed_pred, average="weighted", zero_division=0))

X_train_embed = sample_train["transcript"].apply(get_embedding).tolist()

X_test_embed = sample_test["transcript"].apply(get_embedding).tolist()

print("Embeddings Generated Successfully!")

y_train_embed = sample_train["sentiment"]
y_test_embed = sample_test["sentiment"]

embed_model = LogisticRegression(max_iter=1000)

embed_model.fit(X_train_embed, y_train_embed)

embed_pred = embed_model.predict(X_test_embed)

