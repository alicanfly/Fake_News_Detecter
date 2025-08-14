import joblib

# Load the trained pipeline
pipe = joblib.load('tfidf_logreg_pipeline.joblib')

# Example: Predict on new text
texts = [
    "This is a shocking news headline!",
    "The government announced new policies today."
]
preds = pipe.predict(texts)
probs = pipe.predict_proba(texts)[:, 1]

for t, p, prob in zip(texts, preds, probs):
    print(f"Text: {t}\nPrediction (1=fake, 0=real): {p}, Probability fake: {prob:.3f}\n")