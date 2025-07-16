import yaml
from typing import List, Tuple
import re

def extract_timestamps_and_topics(transcript_entries):
    """
    Extract (time_str, topic) tuples from transcript entries.
    """
    result = []
    for entry in transcript_entries:
        if 'time' not in entry:
            continue
        time_str = str(entry['time'])
        if 'header' in entry and entry['header']:
            topic = entry['header']
        elif 'line' in entry and entry['line']:
            topic = entry['line'].split('.')[0].strip()
        else:
            topic = "No topic"
        result.append((time_str, topic))
    return result

def parse_time_to_seconds(t: str) -> int:
    """
    Convert time string like '0:00' or '1:35' to total seconds.
    """
    parts = list(map(int, t.split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Unexpected time format: {t}")

def chunk_transcript_into_groups(timestamps: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Group timestamps into chunks where consecutive timestamps are <= 5 minutes apart.
    Returns list of (start_time, end_time) tuples.
    """
    if not timestamps:
        return []

    if not (isinstance(timestamps[0], tuple) and isinstance(timestamps[0][0], str)):
        raise ValueError("Input 'timestamps' must be list of tuples (str, str)")

    groups = []
    start_time_str = timestamps[0][0]
    prev_seconds = parse_time_to_seconds(start_time_str)

    for i in range(1, len(timestamps)):
        current_time_str = timestamps[i][0]
        current_seconds = parse_time_to_seconds(current_time_str)

        if current_seconds - prev_seconds > 300:  # more than 5 mins gap
            groups.append((start_time_str, timestamps[i-1][0]))
            start_time_str = current_time_str
        prev_seconds = current_seconds

    groups.append((start_time_str, timestamps[-1][0]))
    return groups

def chunk_text(text: str, max_chars=3000, overlap=200) -> List[str]:
    """
    Split text into chunks of max_chars length with overlap.
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

def main():
    # Load your podcast md/yaml file
    with open('podcast.md', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    transcript = data.get('transcript', [])
    timestamps = extract_timestamps_and_topics(transcript)

    print("Extracted timestamps and topics:")
    for time_str, topic in timestamps:
        print(f"{time_str}: {topic}")

    groups = chunk_transcript_into_groups(timestamps)
    print("\nTimestamp groups (<=5min gap):")
    for start, end in groups:
        print(f"{start} - {end}")

    # Example: chunk full transcript text (concatenate all lines)
    full_text = " ".join(entry.get('line', '') for entry in transcript if 'line' in entry)
    text_chunks = chunk_text(full_text)
    print(f"\nFull transcript split into {len(text_chunks)} chunks (max 3000 chars each):")

    for i, chunk in enumerate(text_chunks[:2], 1):  # print first 2 chunks as example
        print(f"\nChunk {i}:\n{chunk[:500]}...")  # print first 500 chars of chunk

if __name__ == "__main__":
    main()
