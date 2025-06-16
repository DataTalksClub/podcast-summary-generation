from llms.groq_backend import GroqLLM
from pipeline.summarizer import summarize_podcast_full
from utils.io_utils import load_text, save_text
from utils.evaluation import evaluate_summary

def main():
    input_path = "s20e06-from-supply-chain-management-to-digital-warehousing-and-finops.md"
    output_path = "summary_groq_6_16_2024.md"

    podcast_text = load_text(input_path)

    llm = GroqLLM()  # You can easily switch to OpenAI later
    summary, _ = summarize_podcast_full(llm, podcast_text)
    
    save_text(summary, output_path)

    metrics = evaluate_summary(podcast_text, summary)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
