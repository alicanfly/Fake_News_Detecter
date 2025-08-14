# Fake News Detector

This project is a comprehensive solution for detecting fake news using both classical machine learning and state-of-the-art deep learning techniques.

## Features

- **TF-IDF + Logistic Regression Pipeline:** Fast and interpretable baseline.
- **Random Forest Classifier:** Additional classical baseline.
- **BERT Fine-tuning:** Advanced deep learning approach using HuggingFace Transformers (optional, requires GPU).
- **Preprocessing:** Text cleaning, stopword removal, and lemmatization.
- **Evaluation:** Confusion matrix, classification report, ROC-AUC, and ROC curve visualization.
- **Model Saving & Inference:** Easily save and load trained models for predictions.

## Requirements

- Python 3.7+
- Install dependencies:
  ```
  pip install pandas scikit-learn matplotlib seaborn nltk joblib transformers datasets torch tqdm
  ```

## Dataset

- The script expects a CSV file containing news articles and their labels (e.g., `news_combined.csv`).
- If you have separate files for fake and real news, combine them into one file with a `label` column (1 for fake, 0 for real).

## Usage

1. **Prepare your dataset:**  
   Place your combined CSV file in the project directory and update the `DATA_PATH` variable in `fake_news_detection_complete_code.py` if needed.

2. **Run the script:**  
   ```
   python fake_news_detection_complete_code.py
   ```

3. **Results:**  
   - The script will train models, evaluate them, and save the trained models to disk.
   - Sample predictions will be printed at the end.

## Notes

- BERT fine-tuning is optional and requires a compatible GPU and PyTorch with CUDA.
- For best results, ensure your dataset is clean and properly labeled.

## Project Structure

```
.
├── fake_news_detection_complete_code.py
├── news_combined.csv
└── README.md
```

## Author

Task No. 3 of the Project.# Fake_News_Detecter
