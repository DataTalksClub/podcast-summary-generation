def build_prompt(podcast_text):
    return f"""You are an expert podcast editor. Your task is to organize a podcast transcript into logically grouped, thematic segments based on timestamped titles and their content.

### Goal:
Group related, consecutive podcast sections into thematic chunks that make sense as standalone parts of a larger conversation. Each group will be saved as a separate file (e.g., chunk1, chunk2), and should include all the transcript text that falls within that group's time range.

### Input:
You will receive a list of podcast sections. Each section consists of:
- A timestamp (e.g., [00:00])
- A short title summarizing the topic
- A transcript body for that topic

### Format:
[00:00] MLOps in corporations versus startups  
Transcript: Companies of different sizes adopt MLOps differently...

[06:03] The agility and pace of startups  
Transcript: Startups move quickly and iterate fast...

[07:54] MLOps on a shoestring budget  
Transcript: When you don’t have millions to spend on ML infra...

...

### Instructions:
1. Analyze the sequence of timestamped sections.
2. Find related consecutive topics and combine them into logical thematic groups.
3. Each group should:
   - Cover a coherent theme or sub-conversation
   - Stay within a rough range of 5 to 15 minutes, unless the flow clearly continues
4. For each group:
   - List the timestamps and titles
   - Show the time range in the format: Start - End
   - Indicate that the full transcript from that time span will be included in the output file (e.g., chunk1.md, chunk2.md, etc.)

### Output Format:

Group 1  
Timestamps:  
[00:00] MLOps in corporations versus startups  
[06:03] The agility and pace of startups  
[07:54] MLOps on a shoestring budget  
[12:54] Cloud solutions for startups  
Time Range: 00:00 - 15:06  
Includes: chunk1.md

Group 2  
Timestamps:  
[15:06] Platform choices for scaling ML  
[20:15] Vendor lock-in concerns  
[26:44] Hybrid cloud strategies  
Time Range: 15:06 - 30:02  
Includes: chunk2.md

(Continue as needed...)

---

Podcast Sections:  
{podcast_text}
"""
