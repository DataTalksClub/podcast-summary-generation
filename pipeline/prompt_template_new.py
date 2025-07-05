from typing import List, Dict, Tuple
import re

# === Step 1: Content Chunking ===

def chunk_timestamps(timestamp_lines: List[str]) -> List[Dict]:
    """
    Groups consecutive related topics into thematic chunks.

    For demo, this naive example groups every 4 consecutive timestamps.
    You can implement smarter logic here.

    Input example line: "0:00 MLOps in corporations versus startups"
    """
    chunks = []
    group_size = 4  # example grouping size
    for i in range(0, len(timestamp_lines), group_size):
        group = timestamp_lines[i:i+group_size]
        if not group:
            continue

        # Extract timestamps and titles
        timestamps = []
        titles = []
        for line in group:
            m = re.match(r"(\d+:\d+)\s+(.*)", line)
            if m:
                timestamps.append(m.group(1))
                titles.append(line)
        
        chunk = {
            "group_id": len(chunks) + 1,
            "timestamps": titles,
            "start_time": timestamps[0],
            "end_time": timestamps[-1],
            "time_range": f"{timestamps[0]} - {timestamps[-1]}"
        }
        chunks.append(chunk)
    return chunks

# === Step 2: Content Organization ===
# (Here you would save chunk texts separately or just keep them as strings)

def extract_text_by_time_range(full_transcript: str, start: str, end: str) -> str:
    """
    Placeholder: Extract transcript text between start and end times.
    Needs actual logic depending on transcript format.
    """
    # Dummy return: return full transcript for now
    return full_transcript

# === Step 3 & 4: Format Selection and Content Extraction Prompts ===

PROMPTS = {
    "key_takeaways": """
Based on the transcript, extract key takeaways. These should be the most important insights, ideas, or lessons a listener would remember from this section. Avoid vague or generic points. Use clear, specific, high-impact phrasing.
Output as a numbered list. Each item should be ≤25 words.
""",
    "notable_quotes": """
Extract memorable or insightful quotes and their context directly from the transcript. Prefer short, self-contained, high-impact quotes that could stand alone.
""",
    "guest_journey": """
Summarize the guest's career or learning journey as described in this transcript.
Break it into chronological stages using short headings + 1–2 sentence descriptions per stage.
Focus on turning points, obstacles, motivations, and transitions.
""",
    "practical_advice": """
Extract actionable advice or tips mentioned by the guest.
These should be practical, specific, and useful — not vague or motivational.
Format as a list, with each item:
- Tip in bold (max 10 words)
- Supporting sentence or example (optional, max 20 words)
""",
    "resources_mentioned": """
List all specific tools, frameworks, books, libraries, datasets, or products mentioned in the transcript.
For each, provide:
- Name of the resource
- One-sentence description or how it was used/recommended by the guest
"""
}

def extract_content(chunk_text: str, format_type: str) -> str:
    """
    Stub for calling an LLM or custom function with prompt + chunk_text
    Returns a string response.
    """
    prompt = PROMPTS.get(format_type)
    if not prompt:
        raise ValueError(f"Unknown format_type: {format_type}")

    # Combine prompt + transcript chunk as input for extraction model
    full_prompt = prompt + "\nTranscript:\n" + chunk_text

    # Here: Replace with actual call to GPT or another extraction function
    # For demo, just return a placeholder string
    return f"Extracted {format_type} content for chunk (placeholder)."

# === Step 5: Content Consolidation ===

def consolidate_contents(chunks_contents: List[str]) -> str:
    """
    Combine all chunk outputs into a single consolidated string.
    """
    return "\n\n".join(chunks_contents)

# === Step 6: Carousel Generation ===

def generate_carousel_markdown(
    title: str,
    hook: str,
    content_slides: List[Tuple[str, str]],
    cta_title: str,
    cta_text: str
) -> str:
    """
    Generates markdown table for LinkedIn carousel slides.
    content_slides: list of tuples (slide_title, slide_text)
    """
    lines = []
    lines.append("| Slide # | Title | Text |")
    lines.append("|---------|-------|------|")
    # Slide 1: Cover
    lines.append(f"| 1 | {title} | {hook} |")

    # Slides 2-9: content
    for idx, (slide_title, slide_text) in enumerate(content_slides[:8], start=2):
        # Enforce limits (truncate if needed)
        slide_title = slide_title[:10]
        slide_text = slide_text[:120]
        lines.append(f"| {idx} | {slide_title} | {slide_text} |")

    # Slide 10: CTA
    lines.append(f"| 10 | {cta_title[:10]} | {cta_text[:120]} |")

    return "\n".join(lines)

# === Example Usage ===

if __name__ == "__main__":
    # Example input timestamps (your actual input will come from transcript)
    timestamps = [
        "0:00 MLOps in corporations versus startups",
        "6:03 The agility and pace of startups",
        "7:54 MLOps on a shoestring budget",
        "12:54 Cloud solutions for startups",
        "15:06 Challenges of cloud complexity versus on-premise",
        "19:19 Selecting tools and avoiding vendor lock-in",
        "22:22 Choosing between a startup and a corporation",
        "27:30 Flexibility and risks in startups",
        "29:37 Bureaucracy and processes in corporations",
        "33:17 The role of frameworks in corporations",
        "34:32 Advantages of large teams in corporations",
        "40:01 Challenges of technical debt in startups",
        "43:12 Career advice for junior data scientists",
        "44:10 Tools and frameworks for MLOps projects",
        "49:00 Balancing new and old technologies in skill development",
        "55:43 Data engineering challenges and reliability in LLMs",
        "57:09 On-premise vs. cloud solutions in data-sensitive industries",
        "59:29 Alternatives like Dask for distributed systems",
    ]

    # Step 1: Chunk timestamps
    chunks = chunk_timestamps(timestamps)
    for c in chunks:
        print(f"Group {c['group_id']}\nTimestamps:")
        for t in c['timestamps']:
            print(t)
        print(f"Time Range: {c['time_range']}\n")

    # Step 2: Simulate full transcript text (replace with actual transcript)
    full_transcript_text = "Full transcript text here covering all timestamps."

    # Step 3+4: Extract content for each chunk and format
    all_key_takeaways = []
    for c in chunks:
        chunk_text = extract_text_by_time_range(full_transcript_text, c['start_time'], c['end_time'])
        extracted = extract_content(chunk_text, "key_takeaways")
        all_key_takeaways.append(extracted)

    # Step 5: Consolidate
    final_takeaways = consolidate_contents(all_key_takeaways)

    # Step 6: Generate Carousel Markdown
    # Dummy content slides (replace with parsed real content)
    sample_slides = [
        ("Startup Edge", "Startups innovate faster with lean MLOps."),
        ("Cloud Benefits", "Cloud tools reduce DevOps overhead."),
        ("Budget Innovation", "Limited budgets spark creativity."),
        ("Speed Matters", "Quick iterations beat perfect tools."),
    ]
    carousel_md = generate_carousel_markdown(
        title="Key MLOps Insights from Startups",
        hook="Lean teams achieve big impact through speed and agility.",
        content_slides=sample_slides,
        cta_title="Apply These Lessons Today",
        cta_text="Implement these MLOps strategies in your next project."
    )

    print("\n--- Carousel Markdown ---\n")
    print(carousel_md)
