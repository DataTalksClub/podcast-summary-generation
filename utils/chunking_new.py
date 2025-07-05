def chunk_text(text, max_chars=3000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_chars
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def chunk_transcript_into_groups(timestamps):
    """
    Accepts a list of (timestamp, topic) tuples.
    Returns a list of (start_index, end_index) groups for themed segments.
    """
    groups = []
    current_group = []
    start_time = timestamps[0][0]

    for i in range(len(timestamps)):
        current_group.append(timestamps[i])
        # End group if large jump or logical split; placeholder for real logic
        if i == len(timestamps) - 1 or (timestamps[i+1][0] - timestamps[i][0]).seconds > 300:
            end_time = timestamps[i][0]
            groups.append((start_time, end_time))
            if i + 1 < len(timestamps):
                start_time = timestamps[i + 1][0]
                current_group = []
    return groups