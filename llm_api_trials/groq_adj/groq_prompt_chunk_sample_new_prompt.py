import os
from groq import Groq
import os
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer







# ---- Set your API key securely ----
os.environ["GROQ_API_KEY"] = "gsk_" #insert API key here
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found.")

client = Groq(api_key=API_KEY)

# ---- Prompt Template ----
def build_prompt(podcast_text):
    return f"""
You are an expert in summarizing data science and machine learning podcasts, especially those by DataTalksClub.
Given the full podcast transcript:

- Craft a compelling main title that captures the central theme.

- Below the transcript, write a concise one-sentence summary of the entire episode.

- Identify 3–5 key sections of the episode and provide subtitles (each max 8 words) that reflect the flow of the discussion.

- Write 7 highlight points, each a single sentence of no more than 10 words, capturing impactful quotes, tools mentioned, or unique ideas.

- Include a relevant emoji with each highlight to boost engagement.

- Use bullet points for subtitles and highlights.

- Avoid connector words such as “despite,” “however,” “and,” or “moreover.”

- Maintain a professional and engaging tone tailored for LinkedIn carousel posts.

- Do not include hashtags, keywords, or summary takeaways at the end.

Podcast Text:
{podcast_text}
"""

# ---- Chunking Large Input ----
def chunk_text(text, max_chars=3000, overlap=200):
    """
    Splits the input text into overlapping chunks to prevent semantic loss.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_chars
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks

# ---- Summarization API Call ----
def summarize_text(text):
    prompt = build_prompt(text)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gemma2-9b-it",
        #model = "llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content.strip()

# ---- Summarize Full Podcast ----
def summarize_podcast_full(podcast_text):
    chunks = chunk_text(podcast_text)
    partial_summaries = [summarize_text(chunk) for chunk in chunks]

    combined_summary = " ".join(partial_summaries)
    final_summary = summarize_text(combined_summary)

    return final_summary, partial_summaries

# ---- Evaluation Metrics ----
def evaluate_summary(original_text, summary_text):
    vectorizer = TfidfVectorizer().fit_transform([original_text, summary_text])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

    compression_ratio = round(len(summary_text) / len(original_text), 3)

    try:
        import textstat
        readability = textstat.flesch_reading_ease(summary_text)
    except ImportError:
        readability = "Requires textstat library (pip install textstat)"

    return {
        "Compression Ratio": compression_ratio,
        "Coverage Score (cosine similarity)": round(similarity, 3),
        "Readability Score": readability
    }

# ---- File I/O ----
def load_podcast_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_summary_to_file(summary, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

# ---- Optional: Driver Function ----
def main(input_path, output_path):
    podcast_text = load_podcast_text(input_path)
    summary, parts = summarize_podcast_full(podcast_text)
    save_summary_to_file(summary, output_path)

    metrics = evaluate_summary(podcast_text, summary)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

# Example usage:
# main("podcast.txt", "summary.txt")


# Example usage:
# main("podcast.txt", "summary.txt")

#file_path = 'latest_transcript.md'  # Input podcast text file path
#file_path = 'second_latest_transcript.md'  # Input podcast text file path
file_path = 'second_latest_transcript.md'  # Input podcast text file path
file_path  = "third_latest_transcript.md"

#output_file_path = 'podcast_summary_gemma2-9b-it_latest1.txt'
#output_file_path = 'podcast_summary_gemma2-9b-it_latest2.txt'
output_file_path = 'podcast_summary_gemma2-9b-it_latest3.txt'

main(file_path, output_file_path)










