# app.py
from pathlib import Path
import warnings
import joblib
from sklearn.exceptions import InconsistentVersionWarning
import streamlit as st

from src.rules import find_warning_signals

# --------------------------------------------------
# ScamSense: AI-powered SMS spam detection assistant
# --------------------------------------------------

st.set_page_config(
    page_title="ScamSense | SMS Safety Assistant",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "models" / "scamsense_linear_svm.joblib"


@st.cache_resource
def load_model():
    """Loads the serialized scikit-learn TF-IDF + Linear SVM pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model file is missing from the models directory."
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(MODEL_PATH)
    except Exception as err:
        raise RuntimeError(f"Error deserializing model file: {err}") from err


model = None
try:
    model = load_model()
except Exception as exc:
    st.error(
        "⚠️ System Alert: The AI classification model could not be initialized. "
        "Rule-based indicator scanning remains functional."
    )
    with st.expander("Technical Diagnostic"):
        st.write(str(exc))

# --------------------------------------------------
# Sidebar: Academic Evaluation & Measured Benchmarks
# --------------------------------------------------

with st.sidebar:
    st.header("📊 Model Performance")
    st.caption("Evaluated on 1,034 holdout test samples (UCI SMS Collection)")

    st.subheader("Candidate Comparison")
    st.markdown(
        """
        | Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
        | :--- | :---: | :---: | :---: | :---: |
        | **Linear SVM** | **0.9807** | **0.9664** | **0.8779** | **0.9200** |
        | Multinomial NB | 0.9632 | 1.0000 | 0.7099 | 0.8304 |
        | Logistic Reg. | 0.9623 | 1.0000 | 0.7023 | 0.8251 |
        """
    )
    st.caption("Selected: Linear SVM provides the highest balanced spam F1-score.")

    st.divider()

    st.subheader("Holdout Confusion Matrix")
    st.markdown(
        """
        | Actual \\ Pred | Classified Ham | Classified Spam |
        | :--- | :---: | :---: |
        | **Actual Ham** | **899** (TN) | **4** (FP) |
        | **Actual Spam** | **16** (FN) | **115** (TP) |
        """
    )

    st.divider()
    st.caption(
        "ScamSense | First-Semester B.E. AI Applications Lab | Chandigarh University"
    )

# --------------------------------------------------
# Main UI Layout
# --------------------------------------------------

st.title("🛡️ ScamSense")
st.subheader("AI-Powered SMS Spam Detection & Digital Safety Assistant")

st.write(
    "Analyze incoming text messages using machine-learning pattern recognition and heuristic safety indicators."
)

st.info(
    "ScamSense provides an ML-based spam classification and risk pattern flags, "
    "not a guaranteed scam verdict or proof that a message is safe."
)

# Initialize session state for pre-filled sample text
if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""

# Quick-test sample scenarios for evaluators
st.markdown("##### Quick-Test Pre-fills:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💬 Personal Chat", use_container_width=True):
        st.session_state["sms_input"] = (
            "Hey, are we still meeting at the library at 5 PM to review lab notes?"
        )

with col2:
    if st.button("🎁 Prize Scam", use_container_width=True):
        st.session_state["sms_input"] = (
            "Congratulations! You have won a cash prize of $5,000. "
            "Claim your prize immediately at http://claim-award.xyz"
        )

with col3:
    if st.button("🔐 OTP / Account Fraud", use_container_width=True):
        st.session_state["sms_input"] = (
            "URGENT: Your bank account is suspended. Reply with your "
            "OTP and PIN number now to verify identity."
        )

with col4:
    if st.button("💳 Legitimate Bill", use_container_width=True):
        st.session_state["sms_input"] = (
            "Your electric bill of Rs 1,450 was paid successfully on Oct 28. Ref #89321."
        )

message = st.text_area(
    "Paste the SMS you want to analyze",
    value=st.session_state["sms_input"],
    height=150,
    placeholder=(
        "Example: Congratulations! You have won a prize. "
        "Click the link to claim it..."
    ),
    max_chars=5000,
)

analyze = st.button(
    "Analyze message",
    type="primary",
    use_container_width=True,
)

if analyze:
    cleaned_message = message.strip()

    if not cleaned_message:
        st.warning("Please enter a message first.")
    else:
        signals = find_warning_signals(cleaned_message)
        prediction = None

        if model is not None:
            try:
                prediction = model.predict([cleaned_message])[0]
            except Exception as pred_err:
                st.error("An error occurred during machine learning inference.")
                with st.expander("Inference Diagnostic"):
                    st.write(str(pred_err))

        st.divider()
        st.subheader("Analysis Results")

        res_col1, res_col2 = st.columns(2)

        # 1. Model Output Display
        with res_col1:
            st.markdown("#### Model Classification")
            if prediction == "spam":
                st.error("SPAM-LIKE")
                st.write(
                    "The model classified this message as spam. "
                    "Treat unexpected links, offers, and requests for information with caution."
                )
            elif prediction == "ham":
                st.success("LIKELY LEGITIMATE")
                st.write(
                    "The model classified this message as legitimate. "
                    "This does not guarantee that it is safe."
                )
            else:
                st.info("UNAVAILABLE")

        # 2. Separate Rule-based Warning Indicators
        with res_col2:
            st.markdown("#### Warning Indicators")
            if signals:
                for signal in signals:
                    st.warning(signal)
            else:
                st.write(
                    "No indicators from the current rule list were detected. "
                    "This does not mean the message is safe."
                )

        # 3. Defensive Safety Guidance
        st.markdown("### Recommended Safety Steps")
        if signals or prediction == "spam":
            st.markdown(
                "- **Avoid clicking unexpected links.**\n"
                "- **Never share an OTP, password, PIN, or CVV** in response to an unsolicited message.\n"
                "- **Independently verify claims** using the organization's official app, website, or verified phone number.\n"
                "- **Do not pay advance fees** to claim a prize or unblock an account."
            )
        else:
            st.markdown(
                "- **Verify unexpected senders** before taking requested actions.\n"
                "- **Check sensitive requests** through a secondary trusted channel.\n"
                "- **Exercise general caution** before opening external links or downloading attachments."
            )

        with st.expander("How to interpret this result"):
            st.write(
                "The classification is generated by a Linear SVM trained on the SMS "
                "Spam Collection dataset. The warning indicators are separate, "
                "rule-based checks. Neither method can identify every scam, and the "
                "warning list is not exhaustive."
            )

st.caption(
    "ScamSense | Educational project | "
    "Predictions are advisory, not security guarantees."
)