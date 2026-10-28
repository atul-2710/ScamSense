# ScamSense: AI-Powered SMS Spam Detection & Digital Safety Assistant

**ScamSense** is an applied machine learning and defensive digital safety assistant developed for the First-Semester B.E. Computer Science & Engineering AI Applications Lab at Chandigarh University.

The application analyzes SMS text messages using a supervised machine learning pipeline (TF-IDF + Linear Support Vector Machine) while concurrently running an independent, rule-based heuristics engine to highlight suspicious indicators and supply defensive safety recommendations.

---

## 1. System Architecture & Disclaimers

ScamSense operates using a decoupled, two-layer analysis design:

1. **Machine Learning Classifier**: A scikit-learn pipeline predicting whether an inbound message exhibits `SPAM-LIKE` or `LIKELY LEGITIMATE` linguistic patterns.
2. **Rule-Based Indicator Heuristics**: An independent regular expression scanner detecting high-risk threat vectors (suspicious links, artificial urgency, prize wording, credential harvesting, payment demands).

### Critical Distinction & Advisory Scope

- **Not a Phishing/Fraud Guarantee**: The ML model identifies spam patterns learned from mobile SMS corpora; it does not perform live domain reputation checks, network packet inspection, or sender identity verification.
- **Independent Inspection Layers**: Warning indicators are flagged regardless of the ML model's verdict. A message flagged as `LIKELY LEGITIMATE` by the model may still trigger an indicator if it references payment instructions or external links.
- **Non-Guarantee Principle**: ScamSense is an educational decision-support tool. It never asserts that any message is guaranteed safe or definitely fraudulent.

---

## 2. Dataset & Feature Engineering

- **Dataset Source**: [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Original Dataset Volume**: 5,572 messages
  - **Ham (Legitimate)**: 4,825 messages (~86.59%)
  - **Spam**: 747 messages (~13.41%)
- **Data Quality & Preprocessing**:
  - Missing values: 0
  - Duplicate groups identified: 684 total records
  - Exact duplicate records removed: 403 (beyond first occurrences)
  - Identical message text with conflicting labels: 0
- **Cleaned Dataset Volume**: 5,169 unique messages
- **Data Splitting Protocol**:
  - Stratified 80/20 train-test split (`random_state=42`)
  - **Train Set**: 4,135 samples
  - **Test Set**: 1,034 samples
- **TF-IDF Vectorizer Parameters**:
  - `lowercase=True`
  - `strip_accents="unicode"`
  - `ngram_range=(1, 2)` (Unigrams and Bigrams)
  - `min_df=2` (Filters single-occurrence vocabulary noise)
  - `sublinear_tf=True` (Applies $1 + \log(\text{tf})$ scaling to dampen token burstiness)

---

## 3. Candidate Model Comparison & Selection

Three candidate models were evaluated on the identical 20% holdout test set (1,034 unseen messages):

| Model                   | Accuracy   | Spam Precision | Spam Recall | Spam F1-Score | Selection Status         |
| :---------------------- | :--------- | :------------- | :---------- | :------------ | :----------------------- |
| **Linear SVM**          | **0.9807** | **0.9664**     | **0.8779**  | **0.9200**    | **Selected Model**       |
| Multinomial Naive Bayes | 0.9632     | 1.0000         | 0.7099      | 0.8304        | Lower spam recall (~71%) |
| Logistic Regression     | 0.9623     | 1.0000         | 0.7023      | 0.8251        | Lower spam recall (~70%) |

### Selection Rationale

While Multinomial Naive Bayes and Logistic Regression achieved perfect precision (1.0000), they exhibited weak spam recall, missing roughly 29–30% of incoming spam. Linear SVM was selected because it balances high precision (0.9664) with substantially stronger spam recall (0.8779), resulting in a superior Spam F1-score of 0.9200.

---

## 4. Detailed Linear SVM Evaluation Metrics

Evaluated on 1,034 unseen test samples (903 Ham, 131 Spam):

| Class                | Precision | Recall | F1-Score | Support |
| :------------------- | :-------- | :----- | :------- | :------ |
| **Ham (Legitimate)** | 0.9825    | 0.9956 | 0.9890   | 903     |
| **Spam**             | 0.9664    | 0.8779 | 0.9200   | 131     |
| **Macro Average**    | 0.9745    | 0.9367 | 0.9545   | 1034    |
| **Weighted Average** | 0.9805    | 0.9807 | 0.9803   | 1034    |

### Holdout Confusion Matrix Breakdown

```text
                  Predicted Ham    Predicted Spam
Actual Ham             899 (TN)            4 (FP)
Actual Spam             16 (FN)          115 (TP)
```

- **True Positives (TP = 115)**: Spam messages correctly caught.
- **True Negatives (TN = 899)**: Legitimate messages correctly identified.
- **False Positives (FP = 4)**: Legitimate messages incorrectly flagged as spam (e.g., informal calls, personal favors, career service announcements).
- **False Negatives (FN = 16)**: Spam messages missed by the classifier (e.g., adult-content promotions, ringtone subscriptions, informal SMS abbreviations).

---

## 5. Rule-Based Indicator Categories

The independent heuristic engine in `src/rules.py` scans for five high-risk patterns using regular expressions:

1. **Suspicious link**: URLs (`http://`, `https://`, `www.`), link shorteners (`bit.ly`, `tinyurl.com`), and common web suffixes (`.com`, `.net`, `.org`, `.info`, `.xyz`).
2. **Urgency or pressure**: Coercive phrasing (`urgent`, `immediately`, `act now`, `expires today`, `limited time`, `final warning`).
3. **Prize or reward language**: Unsolicited gain phrases (`you've won`, `you have won a prize`, `winner`, `claim your prize`, `free entry`, `lottery`).
4. **Sensitive information request**: Credential-harvesting terms (`password`, `otp`, `one-time password`, `pin number`, `cvv`, `bank details`).
5. **Payment-related language**: Transaction demands (`pay now`, `send money`, `processing fee`, `payment required`, `account suspended`).

---

## 6. Known Limitations & Error Analysis

1. **Absence of Calibrated Probabilities**: The exported Linear SVM classifier utilizes decision hyperplane distances (`decision_function`) rather than calibrated probability estimates (`predict_proba()`). The web interface intentionally avoids displaying fabricated confidence percentages.
2. **False Negatives**: Informal slang, misspelled spam keywords, and novel short-code notifications can evade detection.
3. **False Positives**: Genuine transactional alerts or urgent personal messages containing keywords like "urgent" or "pay" may trigger advisory flags.

---

## 7. Repository Structure

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
│   └── scamsense_linear_svm.joblib    (Excluded from Git via .gitignore)
└── tests/
    ├── test_model.py
    └── test_rules.py
```

---

## 8. Local Setup & Quickstart

### Part 1: Environment Setup & Isolation

Run these commands in your **VS Code terminal on Windows**:

1. **Verify your working directory**:

   ```powershell
   pwd
   ```

   _Confirm you are in the root `ScamSense` directory._

2. **Activate the virtual environment**:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   _(If script execution is restricted on Windows PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or activate via Command Prompt using `.venv\Scripts\activate.bat`)_

3. **Verify Python and Pip paths**:

   ```powershell
   where.exe python
   where.exe pip
   ```

   _Both paths must point to executables inside the `.venv\Scripts\` directory._

4. **Install runtime and test dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

### Part 2: Model Artifact Verification

The trained model pipeline artifact must be present in the `models/` folder:

1. **Check model presence and size**:

   ```powershell
   Get-Item models/scamsense_linear_svm.joblib
   ```

   _Confirm the file exists with a non-zero byte size._

2. **Verify Git ignores the artifact**:
   ```powershell
   git status --ignored
   ```
   _Confirm `models/scamsense_linear_svm.joblib` is listed under "Ignored files", keeping the repository lightweight._

### Part 3: Automated Test Execution

Run the automated test suite to verify the rule engine and model inference contracts:

1. **Execute pytest**:
   ```powershell
   python -m pytest -v
   ```
2. **Verification Suite**:
   - `tests/test_model.py::test_model_loading_and_prediction_contract` -> **PASSED**
   - `tests/test_rules.py::test_normal_personal_message_no_signals` -> **PASSED**
   - `tests/test_rules.py::test_empty_or_none_input` -> **PASSED**
   - `tests/test_rules.py::test_prize_wording_variations` -> **PASSED**
   - `tests/test_rules.py::test_otp_and_sensitive_info_requests` -> **PASSED**
   - `tests/test_rules.py::test_suspicious_urls` -> **PASSED**
   - `tests/test_rules.py::test_urgency_language` -> **PASSED**
   - `tests/test_rules.py::test_payment_related_language` -> **PASSED**
   - `tests/test_rules.py::test_case_insensitive_matching` -> **PASSED**
   - `tests/test_rules.py::test_multi_indicator_trigger` -> **PASSED**

### Part 4: Launching and Testing the Web Dashboard

1. **Launch the Streamlit web application**:

   ```powershell
   python -m streamlit run app.py
   ```

   _Streamlit will host the app at `http://localhost:8501` and open it in your default web browser._

2. **Manual Regression Scenarios**:
   - **Scenario 1 (Personal SMS)**:
     `Hey, are we still heading to the campus cafe after class?`
     _Result_: `LIKELY LEGITIMATE` | _Indicators_: None.
   - **Scenario 2 (Prize Scheme)**:
     `Congratulations! You have won a prize of $5000. Claim your prize at http://claim-award.xyz`
     _Result_: `SPAM-LIKE` | _Indicators_: `Prize or reward language`, `Suspicious link`.
   - **Scenario 3 (Credential Harvesting)**:
     `Urgent: Your account is suspended. Reply immediately with your bank OTP and PIN number.`
     _Result_: `SPAM-LIKE` | _Indicators_: `Urgency or pressure`, `Payment-related language`, `Sensitive information request`.
   - **Scenario 4 (Service/Utility Payment)**:
     `Your electricity bill of Rs 1,450 was received and processed on Oct 28. Ref #89321.`
     _Result_: `LIKELY LEGITIMATE`.
   - **Scenario 5 (Empty Input)**:
     Click **Analyze message** without text.
     _Result_: Prompts `Please enter a message first.`

3. **Terminate Local Server**:
   Press `Ctrl + C` in the VS Code terminal.
