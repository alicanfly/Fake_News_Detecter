# --------------------------
# 1. Imports
# --------------------------
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import joblib

# Optional deep learning imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False

# NLTK
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# 2. Load dataset
# --------------------------

DATA_PATH = 'news_combined.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place your dataset there or change DATA_PATH.")

raw = pd.read_csv(DATA_PATH)
print('Loaded dataset shape:', raw.shape)
print('Columns:', raw.columns.tolist())

# Try to find text and label columns automatically
text_col = None
label_col = None
for c in raw.columns:
    if c.lower() in ['text', 'article', 'content', 'body']:
        text_col = c
    if c.lower() in ['label', 'class', 'truth', 'target']:
        label_col = c

# If label not found, try to infer from common datasets where 'label' isn't explicit
if text_col is None:
    # as fallback, try to use 'title' or first textual column
    for c in raw.columns:
        if raw[c].dtype == object:
            text_col = c
            break

if label_col is None:
    # Some datasets (like the common "Fake and Real News" Kaggle) use filename separation
    # Here we try to create binary label if there's a column 'label' or 'class' else raise error
    # If there's a 'label' mapping in dataset, adjust accordingly.
    if 'label' in raw.columns:
        label_col = 'label'
    elif 'truth' in raw.columns:
        label_col = 'truth'
    else:
        # Try to detect by number of unique values in object columns
        for c in raw.columns:
            if raw[c].dtype == object and raw[c].nunique() <= 10 and c != text_col:
                label_col = c
                break

if label_col is None:
    raise ValueError('Could not automatically detect a label column. Please open the CSV and set LABEL_COL manually in the script.')

print('Using text column:', text_col)
print('Using label column:', label_col)

# Keep only the needed columns
df = raw[[text_col, label_col]].copy()
df = df.dropna().reset_index(drop=True)

# If label is textual e.g., 'FAKE'/'REAL', convert to binary 0/1
if df[label_col].dtype == object:
    df[label_col] = df[label_col].str.lower().map(lambda x: 1 if 'fake' in x or 'false' in x or '0' in str(x) else 0)
else:
    # assume numeric where 1 => fake, 0 => real; if reversed, adjust accordingly
    df[label_col] = df[label_col].astype(int)

# Ensure labels are 0/1
print('Label distribution:')
print(df[label_col].value_counts())

# --------------------------
# 3. Preprocessing
# --------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)  # remove urls
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

# Apply preprocessing (can be slow on big datasets)
print('Preprocessing texts...')
df['clean_text'] = df[text_col].astype(str).apply(preprocess_text)

# --------------------------
# 4. Train/test split
# --------------------------
X = df['clean_text'].values
y = df[label_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print('Train size:', len(X_train), 'Test size:', len(X_test))

# --------------------------
# 5. TF-IDF + Logistic Regression pipeline
# --------------------------
print('\nTraining TF-IDF + Logistic Regression...')
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

pipe.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipe, 'tfidf_logreg_pipeline.joblib')
print('Saved TF-IDF pipeline as tfidf_logreg_pipeline.joblib')

# Predictions
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:,1]

print('\nTF-IDF + LR Evaluation:')
print(classification_report(y_test, y_pred, digits=4))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - TFIDF + LR')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - TFIDF + LR')
plt.show()

# --------------------------
# 6. Random Forest baseline (optional)
# --------------------------
print('\nTraining RandomForest on TF-IDF features (optional)...')
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
joblib.dump((vectorizer, rf), 'tfidf_randomforest.joblib')

y_pred_rf = rf.predict(X_test_tfidf)
y_proba_rf = rf.predict_proba(X_test_tfidf)[:,1]
print('RandomForest Report:')
print(classification_report(y_test, y_pred_rf, digits=4))
print('ROC-AUC RF:', roc_auc_score(y_test, y_proba_rf))

# --------------------------
# 7. BERT Fine-tuning (HuggingFace) - requires HF_AVAILABLE
# --------------------------
if HF_AVAILABLE:
    print('\nStarting BERT fine-tuning (this can be slow and needs GPU)...')
    MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare datasets in HuggingFace "datasets" format
    hf_train = Dataset.from_dict({'text': list(X_train), 'label': list(y_train)})
    hf_test = Dataset.from_dict({'text': list(X_test), 'label': list(y_test)})

    def tokenize_batch(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)

    hf_train = hf_train.map(tokenize_batch, batched=True)
    hf_test = hf_test.map(tokenize_batch, batched=True)

    hf_train = hf_train.rename_column('label', 'labels')
    hf_test = hf_test.rename_column('label', 'labels')
    hf_train.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    hf_test.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir='bert_output',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
        preds = np.argmax(logits, axis=1)
        return {
            'roc_auc': roc_auc_score(labels, probs),
            'precision_recall_f1': classification_report(labels, preds, output_dict=True)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model('bert_finetuned')
    print('BERT fine-tuning complete and saved to bert_finetuned/')
else:
    print('\nHuggingFace Transformers or torch not installed. Skipping BERT section. To enable, install transformers, datasets and torch.')

# --------------------------
# 8. Inference helpers
# --------------------------
def predict_with_tfidf(texts, pipeline=pipe):
    cleaned = [preprocess_text(t) for t in texts]
    preds = pipeline.predict(cleaned)
    probs = pipeline.predict_proba(cleaned)[:,1]
    return preds, probs

if __name__ == '__main__':
    sample = ["Breaking: Scientists discover miracle cure for everything! Read more...", "Parliament passed the new education bill today after long debate."]
    preds, probs = predict_with_tfidf(sample)
    for s, p, prob in zip(sample, preds, probs):
        print('TEXT:', s)
        print('PRED:', p, 'PROB(fake):', prob)
        print('-----')

# End of script
