import re

def preprocess_transcript(md_text: str) -> str:
    # Standardize speaker labels
    text = re.sub(r"\*\*(\w+):\*\*:? ?", r"\1:", md_text)

    # Preserve section headings
    text = re.sub(r"### (.+)", r"\n\n=== \1 ===\n\n", text)

    # Remove markdown formatting artifacts
    text = re.sub(r"[*_`>#\-]{1,3}", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text



