from typing import Optional

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

def build_prompt(text: str, format_type: Optional[str] = None) -> str:
    """
    Constructs a comprehensive prompt combining all content extraction points
    when format_type is None, or a specific prompt if format_type is given.

    Args:
        text: The transcript chunk text.
        format_type: Optional string to select specific prompt type.

    Returns:
        Formatted prompt string ready for LLM consumption.
    """
    if format_type:
        prompt_template = PROMPTS.get(format_type)
        if not prompt_template:
            raise ValueError(f"Unknown format_type: {format_type}")
        prompt = prompt_template.strip()
    else:
        # Combine all prompts into one comprehensive prompt
        prompt_parts = []
        for key, p in PROMPTS.items():
            prompt_parts.append(f"### {key.replace('_', ' ').title()} ###\n{p.strip()}")
        prompt = "\n\n".join(prompt_parts)

    return f"{prompt}\n\nTranscript:\n{text.strip()}"
