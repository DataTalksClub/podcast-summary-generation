import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from llms.base import LLMInterface
from llms.groq_backend import GroqLLM
from llms.openai_backend import OpenAILLM
from pipeline.summarizer import summarize_podcast_full
from utils.evaluation import evaluate_summary
from utils.io_utils import load_text, save_text

LLM: LLMInterface

LLM_PLATFORMS = [
    "OpenAI",
    "Grok",
]

st.set_page_config(page_title="Podcast Summarizer", layout="wide")
st.title("Podcast Summarizer")

# Initialize session state variables
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = False
if "summary_ready" not in st.session_state:
    st.session_state.summary_ready = False
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

uploaded_file = st.file_uploader("Upload your podcast transcript (.md)", type=["md"])

selected_llm_platform = st.selectbox(
    "Choose an LLM Platform",
    options=LLM_PLATFORMS,
    index=0,
)

st.write(f"You selected: {selected_llm_platform}")


def get_env_or_secret(key: str) -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        # If secrets.toml not found or key missing, ignore error silently
        pass
    return os.getenv(key, "")


def get_relevant_api_key_env_name() -> str:
    if selected_llm_platform == "OpenAI":
        return "OPENAI_API_KEY"
    elif selected_llm_platform == "Grok":
        return "GROQ_API_KEY"
    else:
        return ""


api_key_env_name = get_relevant_api_key_env_name()

input_api_key = st.text_input(
    f"Enter your {selected_llm_platform} API key:",
    type="password",
    value=st.session_state.user_api_key,
    help=f"API Key for {selected_llm_platform}. If left blank, we'll try to read {api_key_env_name} from `.env` or `secrets.toml`.",
)

# Persist user input in session state
st.session_state.user_api_key = input_api_key


def api_key_available() -> bool:
    return bool(input_api_key or get_env_or_secret(api_key_env_name))


def summarize_and_display():
    st.session_state.processing_summary = True

    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        tmp_input_path = tmp_input.name

    with st.spinner("Summarizing... please wait."):
        original_text = load_text(tmp_input_path)

        api_key_to_use = input_api_key or get_env_or_secret(api_key_env_name)
        if not api_key_to_use:
            st.warning(f"API key for {selected_llm_platform} is missing. Please enter it above.")
            st.session_state.processing_summary = False
            return

        if selected_llm_platform == "OpenAI":
            LLM = OpenAILLM(api_key=api_key_to_use)
        elif selected_llm_platform == "Grok":
            LLM = GroqLLM(api_key=api_key_to_use)
        else:
            st.error(f"Unsupported LLM platform: {selected_llm_platform}")
            st.session_state.processing_summary = False
            return

        summary, _ = summarize_podcast_full(LLM, original_text)

        if summary:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_output:
                save_text(summary, tmp_output.name)
                tmp_output_path = tmp_output.name

            st.session_state["summary_content"] = summary
            st.session_state["metrics"] = evaluate_summary(original_text, summary)
            st.session_state["download_path"] = tmp_output_path
            st.session_state["summary_ready"] = True

        os.unlink(tmp_input_path)

    st.session_state.processing_summary = False


if uploaded_file is None:
    st.info("Please upload a podcast transcript (.md file) to get started.")
elif not api_key_available():
    st.info(f"Please enter your API key for {selected_llm_platform} above to enable summarization.")
else:
    st.success("Transcript uploaded successfully. Click 'Summarize' to proceed.")

st.button(
    "Summarize",
    key="summarize_button",
    on_click=summarize_and_display,
    disabled=st.session_state.processing_summary or not api_key_available() or uploaded_file is None,
)

if st.session_state.get("summary_ready", False):
    st.subheader("Summary")
    st.markdown(st.session_state["summary_content"])

    # Uncomment below if you want to show evaluation metrics
    # st.subheader("Evaluation Metrics")
    # for key, value in st.session_state["metrics"].items():
    #     st.write(f"{key}: {value}")

    with open(st.session_state["download_path"], "rb") as f:
        st.download_button(
            label="Download Summary",
            data=f,
            file_name="podcast_summary.md",
            mime="text/markdown",
        )
