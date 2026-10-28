# Academic Lab Report: ScamSense — AI-Powered SMS Spam Detection & Digital Safety Assistant

- **Course**: B.E. Computer Science & Engineering — AI Applications Lab
- **Department**: Department of Engineering Foundations (EF-4), Chandigarh University
- **Repository**: [ScamSense GitHub Repository](https://github.com/atul-2710/ScamSense)

---

## 1. Executive Summary

Mobile SMS remains a prevalent communication vector for transaction confirmations, two-factor authentication, and personal messaging. However, its ubiquitous nature makes it a frequent target for unsolicited marketing and fraudulent schemes. **ScamSense** is an applied machine learning and digital safety assistant developed to classify SMS text messages while providing real-time, rule-based indicators and defensive guidance.

The system implements a dual-layer architectural approach:

1. A machine learning pipeline utilizing Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction paired with a Linear Support Vector Machine (Linear SVM) trained on the UCI SMS Spam Collection dataset.
2. A rule-based pattern matching module that scans for high-risk text characteristics (e.g., suspicious URLs, artificial urgency, prize wording, sensitive credential harvesting, and payment demands).

Evaluated on 1,034 unseen test records, the Linear SVM classifier achieved an overall accuracy of **98.07%**, a spam precision of **96.64%**, and a spam recall of **87.79%** ($F_1$-score: **0.9200**), outperforming candidate Naive Bayes and Logistic Regression baselines on spam detection coverage.

---

## 2. Problem Statement & Design Objectives

### 2.1 Problem Definition

Automated text classification in SMS presents unique challenges:

- Messages are short, highly informal, and frequently contain non-standard abbreviations, typographical variations, and irregular punctuation.
- Class imbalance is pronounced; benign personal and transactional messages vastly outnumber unsolicited spam.
- Pure statistical classifiers lack explainability and may miss high-risk signals embedded in novel phrasing. Conversely, pure rule-based filters are brittle and easily circumvented by minor lexical modifications.

### 2.2 System Design Objectives

1. **Model Generalization**: Attain high precision to prevent legitimate transactional and personal SMS from being falsely categorized as spam, while maintaining strong recall.
2. **Defensive Separation of Concerns**: Decouple spam classification from risk indicator detection. A message classified as legitimate must still be flagged if it solicits credentials or payments.
3. **Non-Guarantee Principle**: Present classifications transparently as advisory assessments rather than absolute guarantees of safety or fraud.

---

## 3. Dataset Characteristics & Preprocessing

### 3.1 Dataset Overview

The model was developed using the public **UCI SMS Spam Collection** corpus:

- **Total Ingested Samples**: 5,572 messages
- **Initial Class Distribution**: 4,825 Ham (~86.59%) and 747 Spam (~13.41%)
- **Data Integrity Check**: Zero missing values detected. No instances of identical message texts mapped to conflicting labels were found.

### 3.2 Deduplication

Analysis identified 684 messages belonging to duplicate groups. To prevent data leakage between training and testing sets, exact duplicate records beyond their first occurrence were removed (403 records dropped). The final cleaned dataset comprises **5,169 unique messages**.

### 3.3 Data Splitting Protocol

The cleaned dataset was partitioned using an 80/20 train-test split:

- **Splitting Strategy**: Stratified sampling (`stratify=y`) with `random_state=42`.
- **Training Subset**: 4,135 samples (3,539 Ham, 596 Spam).
- **Holdout Test Subset**: 1,034 samples (903 Ham, 131 Spam).

### 3.4 Feature Engineering (TF-IDF Vectorization)

Text data was transformed into numerical sparse matrices using `TfidfVectorizer` configured with:

- `lowercase=True` & `strip_accents="unicode"`: Normalizes character casing and accent variations.
- `ngram_range=(1, 2)`: Captures both individual tokens and contiguous bi-grams (e.g., "win cash", "urgent reply").
- `min_df=2`: Removes tokens appearing in fewer than two records to mitigate sparse vocabulary noise.
- `sublinear_tf=True`: Applies logarithmic term-frequency scaling ($1 + \log(\text{tf})$) to prevent high-frequency words from disproportionately weighting feature vectors.

---

## 4. Model Selection & Experimental Results

Three candidate architectures were trained on the identical 80% training split and evaluated on the 1,034 holdout samples.

### 4.1 Candidate Benchmark

| Model Candidate                            | Test Accuracy | Spam Precision | Spam Recall | Spam $F_1$-Score |
| :----------------------------------------- | :-----------: | :------------: | :---------: | :--------------: |
| **Linear SVM (`LinearSVC`)**               |  **0.9807**   |   **0.9664**   | **0.8779**  |    **0.9200**    |
| Multinomial Naive Bayes (`MultinomialNB`)  |    0.9632     |     1.0000     |   0.7099    |      0.8304      |
| Logistic Regression (`LogisticRegression`) |    0.9623     |     1.0000     |   0.7023    |      0.8251      |

### 4.2 Selection Rationale

While Multinomial Naive Bayes and Logistic Regression achieved perfect precision (1.0000) on the test split, they exhibited an unacceptable false-negative rate, failing to detect ~29–30% of incoming spam messages. Linear SVM was selected as the production model because it achieved a substantially stronger recall rate of **87.79%** while maintaining an exceptional precision of **96.64%** and the highest overall $F_1$-score (**0.9200**).

### 4.3 Detailed Linear SVM Performance

| Class                | Precision | Recall | $F_1$-Score | Support |
| :------------------- | :-------: | :----: | :---------: | :-----: |
| **Ham (Legitimate)** |  0.9825   | 0.9956 |   0.9890    |   903   |
| **Spam**             |  0.9664   | 0.8779 |   0.9200    |   131   |
| **Macro Average**    |  0.9745   | 0.9367 |   0.9545    |  1,034  |
| **Weighted Average** |  0.9805   | 0.9807 |   0.9803    |  1,034  |

### 4.4 Holdout Confusion Matrix Analysis

- **True Positives (TP)**: 115 spam messages correctly identified.
- **True Negatives (TN)**: 899 legitimate messages correctly classified.
- **False Positives (FP)**: 4 legitimate messages incorrectly flagged as spam (0.44% error rate on ham).
- **False Negatives (FN)**: 16 spam messages missed (12.21% error rate on spam).

---

## 5. Error Analysis & Limitations

### 5.1 Qualitative Inspection of Classification Errors

1. **Missed Spam (False Negatives)**:
   - Spam messages relying on short, conversational language or heavy slang (e.g., "dating chat" promos, ringtone delivery confirmations).
   - Messages lacking overt commercial keywords where spam intent relies entirely on external telephony infrastructure (e.g., "Missed call from 087... Call back now").
2. **Spam False Alarms (False Positives)**:
   - Academic and career services notices containing urgency or external contact prompts.
   - Informal personal favors requesting immediate callbacks.

### 5.2 Technical & Architectural Limitations

- **Probability Calibration**: Linear SVM models compute a geometric distance from the separating hyperplane (`decision_function`) rather than calibrated probabilistic confidence (`predict_proba`). To avoid misleading users, the user interface does not display synthetic percentage scores.
- **Contextual Awareness**: The model evaluates isolated SMS texts without access to broader metadata (e.g., sender reputation, telecom routing details, network origin).

---

## 6. Defensive Rule-Based Module Architecture

To provide explainable safety indicators that complement the statistical model, an independent regular expression heuristic engine was built into `src/rules.py`.

The module scans for five distinct risk indicators:

1. **Suspicious Link**: Matches HTTP/HTTPS protocols, shortener domains (`bit.ly`, `tinyurl.com`), and common phishing top-level domains (`.xyz`, `.info`).
2. **Urgency or Pressure**: Detects coercive temporal language (`urgent`, `immediately`, `act now`, `final warning`).
3. **Prize or Reward Language**: Identifies unsolicited incentive phrases (`you've won`, `won a prize`, `winner`, `lottery`).
4. **Sensitive Information Request**: Scans for credential-harvesting keywords (`otp`, `one-time password`, `cvv`, `pin number`, `password`).
5. **Payment-Related Language**: Flags transaction demands (`pay now`, `send money`, `processing fee`, `account suspended`).

---

## 7. Software Verification & Testing Methodology

Automated software testing was integrated using `pytest` to guarantee deterministic behavior across module boundaries.

### 7.1 Test Suite Breakdown

1. `tests/test_rules.py`:
   - Validates that legitimate personal messages return an empty signal list.
   - Asserts pattern capture for single-category indicators (links, urgency, prizes, credentials, payments).
   - Evaluates multi-indicator triggering and case-insensitivity.
   - Tests edge cases including empty strings and non-string inputs.
2. `tests/test_model.py`:
   - Verifies the integrity of the serialized scikit-learn pipeline (`scamsense_linear_svm.joblib`).
   - Asserts the prediction contract over multiple test inputs to ensure output labels belong to the set `{"ham", "spam"}`.

All 10 automated test specifications pass consistently in the local runtime environment.

---

## 8. Conclusion & Future Roadmap

ScamSense demonstrates how pairing supervised text classification with explicit, rule-based indicators creates a robust, user-centric safety assistant. By prioritizing balanced precision and recall, the tool avoids excessive false alarms on legitimate communications while flagging high-risk text patterns.

### Future Work

1. **Transformer-Based Embeddings**: Benchmarking small distilled language models (e.g., DistilBERT) against Linear SVM to improve semantic context retention on informal slang.
2. **Active Threat Intelligence**: Incorporating real-time URL reputation lookup APIs to assess link destinations dynamically.
3. **Sender Domain Verification**: Extending input parameters to analyze sender ID patterns (e.g., alphanumeric headers vs. short codes).
