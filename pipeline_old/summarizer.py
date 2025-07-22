from utils.chunking.chunking import chunk_text

def summarize_podcast_full(llm, podcast_text):
    chunks = chunk_text(podcast_text)
    partial_summaries = [llm.summarize(chunk) for chunk in chunks]
    combined_summary = " ".join(partial_summaries)
    final_summary = llm.summarize(combined_summary)
    return final_summary, partial_summaries
