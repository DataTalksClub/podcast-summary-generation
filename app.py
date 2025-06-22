import streamlit as st
import tempfile
import os

from llms.openai_backend import OpenAILLM
from pipeline.summarizer import summarize_podcast_full
from utils.io_utils import load_text, save_text
from utils.evaluation import evaluate_summary

st.set_page_config(page_title="Podcast Summarizer", layout="wide")
st.title("Podcast Summarizer")

# Upload markdown transcript
uploaded_file = st.file_uploader("Upload your podcast transcript (.md)", type=["md"])

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        tmp_input_path = tmp_input.name

    st.success("Transcript uploaded successfully. Click 'Summarize' to proceed.")

    if st.button("Summarize"):
        # Load the uploaded markdown content
        original_text = load_text(tmp_input_path)

        # Use your LLM summarization pipeline
        llm = OpenAILLM()
        summary, _ = summarize_podcast_full(llm, original_text)

        # Save summary to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_output:
            save_text(summary, tmp_output.name)
            tmp_output_path = tmp_output.name

        # Display summary
        st.subheader("Summary")
        st.markdown(summary)

        # Show evaluation metrics
        st.subheader("Evaluation Metrics")
        metrics = evaluate_summary(original_text, summary)
        for key, value in metrics.items():
            st.write(f"{key}: {value}")

        # Provide download button
        with open(tmp_output_path, "rb") as f:
            st.download_button(
                label="Download Summary",
                data=f,
                file_name="podcast_summary.md",
                mime="text/markdown"
            )

        # Clean up temp files
        os.unlink(tmp_input_path)
        os.unlink(tmp_output_path)
