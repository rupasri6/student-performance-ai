from src.app.streamlit_app import DEFAULTS, pages
from src.data.validate import validate_inference_payload


def test_default_form_payload_is_valid():
    assert validate_inference_payload(DEFAULTS) == []


def test_app_pages_registered():
    assert set(pages) >= {"Home", "Predict", "Performance", "Admin", "Ethics"}
