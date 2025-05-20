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
