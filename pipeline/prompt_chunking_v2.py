def build_prompt(podcast_text):
    return f"""
You are an expert podcast editor. Your task is to organize a podcast transcript into **logically grouped, thematic segments** based on timestamped titles and their content.

### Goal:
Group related, consecutive podcast sections into **thematic chunks** that make sense as standalone parts of a larger conversation. Each group will be saved as a separate file (e.g., chunk1, chunk2), and should include all the transcript text that falls under that thematic umbrella.

### Instructions:
1. **Identify main themes** or topics in the transcript by analyzing titles and content.
2. **Group consecutive segments** that revolve around the same or similar topics.
3. For each group:
   - Give a **short, descriptive title** (2–6 words).
   - Include **start and end timestamps** (based on the earliest and latest timestamps in that group).
   - List all the content under those timestamps **as-is** (don't paraphrase).
4. The chunks should not overlap and should cover the **entire transcript**.

### Example Format:

**Chunk Title:** The Future of AI  
**Start - End:** 00:12:05 - 00:28:33  
**Transcript:**  
[00:12:05] Host: Let’s talk about the future of AI...  
[00:14:12] Guest: One exciting development is...  
...  
[00:28:33] Host: That wraps up our segment on AI.

Repeat for each theme you find.

### Transcript:
{podcast_text}
"""
