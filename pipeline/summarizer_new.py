from utils.chunking_new import chunk_text, chunk_transcript_into_groups

def summarize_podcast_full(llm, podcast_text, timestamps=None):
    if timestamps:
        groups = chunk_transcript_into_groups(timestamps)
        grouped_chunks = [podcast_text[start:end] for (start, end) in groups]
    else:
        grouped_chunks = chunk_text(podcast_text)

    partial_summaries = [llm.summarize(chunk) for chunk in grouped_chunks]
    combined_summary = " ".join(partial_summaries)
    final_summary = llm.summarize(combined_summary)
    return final_summary, partial_summaries
