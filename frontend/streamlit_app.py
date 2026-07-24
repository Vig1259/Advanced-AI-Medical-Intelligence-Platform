"""
Streamlit frontend for the Advanced AI Medical Intelligence Platform.

Run:
    streamlit run frontend/streamlit_app.py

Expects the FastAPI backend to be running (default http://localhost:8000).
Set API_BASE_URL env var to point elsewhere (e.g. a deployed backend).
"""
import base64
import inspect
import os
from pathlib import Path

import requests
import streamlit as st

# set_page_config MUST be the very first Streamlit command in the script.
st.set_page_config(page_title="AI Medical Intelligence Platform", page_icon="🩻", layout="wide")


def _secrets_file_exists() -> bool:
    """
    Only the two paths Streamlit itself checks. We test for the file's
    existence directly, rather than touching st.secrets and catching the
    resulting exception -- st.secrets prints its own "No secrets found"
    notice the moment it's accessed at all, even inside a try/except, so
    catching the exception doesn't suppress the message. Checking the file
    first avoids ever touching st.secrets when there's nothing to load.
    """
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidates)


API_BASE_URL = os.environ.get("API_BASE_URL")
if not API_BASE_URL:
    if _secrets_file_exists():
        # Streamlit Community Cloud exposes deploy-time config via st.secrets
        # (Settings -> Secrets in the app dashboard) rather than OS env vars.
        API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
    else:
        API_BASE_URL = "http://localhost:8000"

# --- Version-compatibility shims ---
_IMAGE_SUPPORTS_CONTAINER_WIDTH = "use_container_width" in inspect.signature(st.image).parameters
_BUTTON_SUPPORTS_CONTAINER_WIDTH = "use_container_width" in inspect.signature(st.button).parameters


def show_image(data, caption=None):
    if _IMAGE_SUPPORTS_CONTAINER_WIDTH:
        st.image(data, caption=caption, use_container_width=True)
    else:
        st.image(data, caption=caption)


def show_button(label, **kwargs):
    if not _BUTTON_SUPPORTS_CONTAINER_WIDTH:
        kwargs.pop("use_container_width", None)
    return st.button(label, **kwargs)


# --- Auth state ---
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None


def auth_headers() -> dict:
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def do_login(username: str, password: str):
    try:
        # /auth/login expects OAuth2 form data (username/password fields),
        # not JSON -- this matches FastAPI's OAuth2PasswordRequestForm.
        resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=15,
        )
        if resp.status_code == 200:
            token_data = resp.json()
            st.session_state.access_token = token_data["access_token"]
            st.session_state.username = username
            st.success("Logged in.")
            st.rerun()
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Login failed: {detail}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")


def do_register(username: str, password: str):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"username": username, "password": password},
            timeout=15,
        )
        if resp.status_code == 201:
            st.success("Account created. You can now log in below.")
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Registration failed: {detail}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")


# --- Login gate: nothing else renders until authenticated ---
if not st.session_state.access_token:
    st.title("🩻 Advanced AI Medical Intelligence Platform")
    st.caption("Please log in or create an account to continue.")

    login_tab, register_tab = st.tabs(["Log In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In", type="primary"):
                if login_username and login_password:
                    do_login(login_username, login_password)
                else:
                    st.warning("Enter both a username and password.")

    with register_tab:
        with st.form("register_form"):
            reg_username = st.text_input("Choose a username", key="reg_username")
            reg_password = st.text_input(
                "Choose a password (min 8 characters)", type="password", key="reg_password"
            )
            if st.form_submit_button("Create Account", type="primary"):
                if reg_username and reg_password:
                    do_register(reg_username, reg_password)
                else:
                    st.warning("Enter both a username and password.")

    st.stop()  # halt rendering here until logged in

# --- Everything below only renders once authenticated ---
with st.sidebar:
    st.write(f"Logged in as **{st.session_state.username}**")
    if show_button("Log Out", use_container_width=True):
        st.session_state.access_token = None
        st.session_state.username = None
        st.rerun()

st.title("🩻 Advanced AI Medical Intelligence Platform")
st.caption("Chest X-Ray Pneumonia Detection · Grad-CAM Explainability · AI-Assisted Reporting")

st.warning(
    "⚠️ Research/demo system only. Not a certified medical device. "
    "All outputs must be reviewed by a licensed physician before any clinical use.",
    icon="⚠️",
)

tab_predict, tab_history = st.tabs(["🔍 New Analysis", "📜 Prediction History"])

with tab_predict:
    col_upload, col_results = st.columns([1, 1.3])

    with col_upload:
        st.subheader("Upload Chest X-Ray")
        uploaded_file = st.file_uploader("Choose a JPEG or PNG image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            show_image(uploaded_file, caption="Uploaded X-Ray")
            run_button = show_button("Run Analysis", type="primary", use_container_width=True)
        else:
            run_button = False

    with col_results:
        st.subheader("Results")
        if uploaded_file and run_button:
            with st.spinner("Running model inference, Grad-CAM, and generating report..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE_URL}/predict", files=files, headers=auth_headers(), timeout=60
                    )
                    if response.status_code == 401:
                        st.error("Session expired. Please log in again.")
                        st.session_state.access_token = None
                        st.rerun()
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to reach the API: {e}")
                    result = None

            if result:
                predicted = result["predicted_class"]
                confidence = result["confidence"]

                badge_color = "red" if predicted == "PNEUMONIA" else "green"
                st.markdown(f"### Prediction: :{badge_color}[{predicted}]")
                st.metric("Confidence", f"{confidence:.1%}")

                st.write("**Class probabilities:**")
                st.bar_chart(result["class_probabilities"])

                st.write("**Grad-CAM Explanation:**")
                gradcam_bytes = base64.b64decode(result["gradcam_image_base64"])
                show_image(gradcam_bytes, caption="Regions influencing the prediction (Grad-CAM)")

                st.write("**AI-Assisted Draft Report:**")
                st.info(result["llm_report"])

                st.caption(f"Record ID: `{result['id']}` · {result['created_at']}")
        elif not uploaded_file:
            st.info("Upload an X-ray image and click 'Run Analysis' to see results here.")

with tab_history:
    st.subheader("Prediction History")
    st.caption("Showing your predictions only.")
    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        class_filter = st.selectbox("Filter by class", ["All", "NORMAL", "PNEUMONIA"])
    with col_refresh:
        st.write("")
        refresh = show_button("Refresh", use_container_width=True)

    try:
        params = {"limit": 50}
        if class_filter != "All":
            params["predicted_class"] = class_filter
        resp = requests.get(f"{API_BASE_URL}/history", params=params, headers=auth_headers(), timeout=30)
        if resp.status_code == 401:
            st.error("Session expired. Please log in again.")
            st.session_state.access_token = None
            st.rerun()
        resp.raise_for_status()
        records = resp.json()

        if records:
            for rec in records:
                with st.expander(f"{rec['predicted_class']} ({rec['confidence']:.1%}) — {rec['original_filename']} — {rec['created_at']}"):
                    detail_resp = requests.get(
                        f"{API_BASE_URL}/history/{rec['id']}", headers=auth_headers(), timeout=30
                    )
                    if detail_resp.ok:
                        detail = detail_resp.json()
                        st.json(detail["class_probabilities"])
                        if detail.get("llm_report"):
                            st.write(detail["llm_report"])
        else:
            st.info("No predictions recorded yet.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load history: {e}")