from utils.chunking.chunking_new import chunk_text, chunk_transcript_into_groups, parse_time_to_seconds

def extract_text_by_time_range(transcript, start_time, end_time) -> str:
    """Extract concatenated transcript lines where time is between start_time and end_time."""
    start_sec = parse_time_to_seconds(start_time)
    end_sec = parse_time_to_seconds(end_time)

    selected_lines = []
    for entry in transcript:
        if 'time' not in entry or 'line' not in entry:
            continue
        line_sec = parse_time_to_seconds(str(entry['time']))
        if start_sec <= line_sec <= end_sec:
            selected_lines.append(entry['line'])

    return " ".join(selected_lines)

def summarize_podcast_full(llm, podcast_text, timestamps=None, transcript=None):
    """
    Summarize podcast text chunked either by timestamp groups or by fixed-size chunks.

    Parameters:
    - llm: object with method summarize(text: str) -> str
    - podcast_text: full transcript text as a string
    - timestamps: optional list of (time_str, topic) tuples
    - transcript: full transcript data (list of dict entries), needed if timestamps used

    Returns:
    - final_summary: combined summary string
    - partial_summaries: list of summaries for each chunk
    """
    if timestamps and transcript:
        groups = chunk_transcript_into_groups(timestamps)
        grouped_chunks = []
        for start, end in groups:
            text_chunk = extract_text_by_time_range(transcript, start, end)  # renamed variable
            if text_chunk.strip():
                grouped_chunks.append(text_chunk)
            else:
                # fallback to full text chunk if no lines found
                grouped_chunks.append(podcast_text)
    else:
        # chunk by fixed size if no timestamps provided
        grouped_chunks = chunk_text(podcast_text)

    partial_summaries = [llm.summarize(chunk) for chunk in grouped_chunks]
    combined_summary = " ".join(partial_summaries)
    final_summary = llm.summarize(combined_summary)

    return final_summary, partial_summaries
