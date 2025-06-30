import os
import tempfile

import streamlit as st

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

uploaded_file = st.file_uploader("Upload your podcast transcript (.md)", type=["md"])

selected_llm_platform = st.selectbox(
    "Choose an LLM Platform",
    options=LLM_PLATFORMS,
    index=0,
)

st.write(f"You selected: {selected_llm_platform}")

input_api_key = st.text_input(
    "Enter your API key (Optional):",
    type="password",
    help="An API Key for the relevant model. If this is empty, an attempt will be made to detect the API keys (OPEN_API_KEY, GROK_API_KEY) from the environmental variables",
)

# Initialize session state
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = False
if "summary_ready" not in st.session_state:
    st.session_state.summary_ready = False


def summarize_and_display():
    st.session_state.processing_summary = True

    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        tmp_input_path = tmp_input.name

    with st.spinner("Summarizing... please wait."):
        original_text = load_text(tmp_input_path)

        if selected_llm_platform == "OpenAI":
            LLM = OpenAILLM(api_key=input_api_key or st.secrets.get("OPENAI_API_KEY"))
        elif selected_llm_platform == "Grok":
            LLM = GroqLLM(api_key=input_api_key or st.secrets.get("GROK_API_KEY"))

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


if uploaded_file is not None:
    st.success("Transcript uploaded successfully. Click 'Summarize' to proceed.")

    st.button(
        "Summarize",
        key="summarize_button",
        on_click=summarize_and_display,
        disabled=st.session_state.processing_summary,
    )

if st.session_state.get("summary_ready", False):
    st.subheader("Summary")
    st.markdown(st.session_state["summary_content"])

    # Metrics are computed but not shown
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
