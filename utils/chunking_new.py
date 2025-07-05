def chunk_text(text, max_chars=3000, overlap=200):
    """
    Split text into chunks of max_chars length with given overlap.
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


def chunk_transcript_into_groups(timestamps):
    """
    Accepts a list of (timestamp, topic) tuples.
    Returns a list of (start_time, end_time) tuples grouping related timestamps.
    
    NOTE: timestamps[i][0] should be a datetime or comparable numeric time.
    This example assumes timestamps are datetime.timedelta or datetime objects for subtraction.
    """
    groups = []
    start_time = timestamps[0][0]
    current_group = [timestamps[0]]

    for i in range(1, len(timestamps)):
        current_time = timestamps[i][0]
        prev_time = timestamps[i - 1][0]
        # If gap > 5 minutes, start a new group
        if (current_time - prev_time).total_seconds() > 300:
            end_time = prev_time
            groups.append((start_time, end_time))
            start_time = current_time
            current_group = []
        current_group.append(timestamps[i])

    # Append last group
    groups.append((start_time, current_group[-1][0]))
    return groups
