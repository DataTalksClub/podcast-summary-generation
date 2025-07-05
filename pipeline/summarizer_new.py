from utils.chunking_new import chunk_text, chunk_transcript_into_groups

def summarize_podcast_full(llm, podcast_text, timestamps=True):
    """
    Summarize the full podcast text by chunking it either by timestamps or fixed chunks,
    then summarize each chunk and finally summarize the combined summaries.
    
    Parameters:
        llm: object with .summarize(text: str) -> str method
        podcast_text: full transcript as a single string
        timestamps: Optional list of (timestamp, topic) tuples or None

    Returns:
        final_summary: str, final combined summary
        partial_summaries: list of str, partial chunk summaries
    """
    if timestamps:
        # chunk_transcript_into_groups expects timestamps with datetime or numeric timestamps,
        # so your timestamps should be parsed accordingly before passing here.
        groups = chunk_transcript_into_groups(timestamps)
        grouped_chunks = []
        for start, end in groups:
            # Here you need logic to extract substring from podcast_text by start/end times,
            # but since your podcast_text is a string, it requires some parsing (e.g., via regex or line indices).
            # Placeholder: just append full text for now (you can implement actual extraction)
            grouped_chunks.append(podcast_text) 
    else:
        grouped_chunks = chunk_text(podcast_text)

    partial_summaries = [llm.summarize(chunk) for chunk in grouped_chunks]
    combined_summary = " ".join(partial_summaries)
    final_summary = llm.summarize(combined_summary)
    return final_summary, partial_summaries
