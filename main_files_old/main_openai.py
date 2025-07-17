import argparse
from llms.openai_backend_old import OpenAILLM
from pipeline.summarizer import summarize_podcast_full
from utils.io_utils import load_text, save_text
from utils.evaluation import evaluate_summary

def main():
    parser = argparse.ArgumentParser(description="Summarize a Markdown podcast transcript using Groq LLM")
    parser.add_argument("--input", "-i", required=True, help="Path to the input .md file (podcast transcript)")
    parser.add_argument("--output", "-o", required=True, help="Path to the output .md file (summary)")
    args = parser.parse_args()

    if not args.input.endswith(".md") or not args.output.endswith(".md"):
        raise ValueError("Both input and output files must be in Markdown (.md) format.")

    # Load the Markdown transcript
    podcast_text = load_text(args.input)

    # Generate the summary
    llm = OpenAILLM()
    summary, _ = summarize_podcast_full(llm, podcast_text)

    # Save the summary as Markdown
    save_text(summary, args.output)

    # Evaluate summary and print metrics
    metrics = evaluate_summary(podcast_text, summary)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
