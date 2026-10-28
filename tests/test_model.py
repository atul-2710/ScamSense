# tests/test_model.py
from pathlib import Path
import warnings
import joblib
import pytest
from sklearn.exceptions import InconsistentVersionWarning

MODEL_PATH = Path("models/scamsense_linear_svm.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model artifact not present locally")
def test_model_loading_and_prediction_contract():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict"), "Model artifact does not implement predict()"

    test_messages = [
        "Hey, can you pick up some groceries on your way back?",
        "URGENT! You have won a 1000 cash prize! Call 09061701461 to claim.",
    ]
    predictions = model.predict(test_messages)
    assert len(predictions) == 2
    assert set(predictions).issubset({"ham", "spam"})