# ScamSense: AI-Powered SMS Spam Detection & Digital Safety Assistant

**ScamSense** is an applied machine learning and defensive safety tool developed for the AI Applications Lab (Department of Engineering Foundations, Chandigarh University). The application analyzes inbound SMS messages using a supervised machine learning pipeline and evaluates risk patterns using rule-based indicator heuristics.

---

## Key System Architecture & Disclaimers

1. **Machine Learning Pipeline**: Uses TF-IDF vectorization paired with a Linear Support Vector Machine (Linear SVM) trained on the UCI SMS Spam Collection dataset to classify messages as either `SPAM-LIKE` or `LIKELY LEGITIMATE`.
2. **Rule-Based Warning Indicators**: Operates an independent pattern-scanning module detecting high-risk language including suspicious links, artificial urgency, prize claims, credential harvesting (OTPs, passwords), and payment demands.
3. **Important Safety Distinction**:
   - The ML model classifies general spam vs. ham; it is not a complete scam or phishing detection engine.
   - The rule-based checks highlight risk factors independently of the ML decision.
   - **Non-Guarantee Principle**: The tool never asserts that any message is guaranteed safe or definitely fraudulent.

---

## Measured Model Performance

The Linear SVM pipeline was evaluated on a stratified 20% holdout test set (1,034 samples) from the UCI SMS Spam Collection (after deduplication):

| Metric        | Ham (Legitimate) | Spam   | Overall / Weighted Avg |
| :------------ | :--------------- | :----- | :--------------------- |
| **Precision** | 0.9825           | 0.9664 | 0.9805                 |
| **Recall**    | 0.9956           | 0.8779 | 0.9807                 |
| **F1-Score**  | 0.9890           | 0.9200 | 0.9803                 |

**Holdout Confusion Matrix Counts**:

- True Negatives (Ham correctly identified): 899
- True Positives (Spam correctly identified): 115
- False Positives (Legitimate flagged as spam): 4
- False Negatives (Spam missed): 16

---

## Repository Structure

```text
ScamSense/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── data/
│   └── raw/
│       └── README.md
├── notebooks/
│   └── ScamSense_ML_Development.ipynb
├── src/
│   └── rules.py
├── models/
│   └── scamsense_linear_svm.joblib  (Excluded from version control)
└── tests/
    ├── test_model.py
    └── test_rules.py
```
