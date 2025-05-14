# Podcast Summary Generation Workflow

This document outlines a potential workflow for generating engaging social media content from podcast transcripts, with a focus on LinkedIn carousels.

## Workflow Steps

### Step 1: Content Chunking
Group related timestamps into larger thematic chunks for better content organization.

**Prompt:**
```
Review the timestamp outline and group related topics into larger chunks.

Instructions:
1. Find related consecutive topics
2. Combine them into logical groups
3. Note start time of first topic and end time of last topic in group

Output Format:
Group 1
Timestamps:
0:00 MLOps in corporations versus startups
6:03 The agility and pace of startups
7:54 MLOps on a shoestring budget
12:54 Cloud solutions for startups
Time Range: 0:00 - 15:06

Group 2
[...]
```

Each group would be {`chunk_1`, `chunk_2`, ..., `chunk_n`} - a set of files or data elements which would contain a specific part of the transcript specified in a relevant time range.

Example input from [MLOps in Corporations and Startups](https://datatalks.club/podcast/s20e04-mlops-in-corporations-and-startups.html) podcast:

```
0:00 MLOps in corporations versus startups
6:03 The agility and pace of startups
7:54 MLOps on a shoestring budget
12:54 Cloud solutions for startups
15:06 Challenges of cloud complexity versus on-premise
19:19 Selecting tools and avoiding vendor lock-in
22:22 Choosing between a startup and a corporation
27:30 Flexibility and risks in startups
29:37 Bureaucracy and processes in corporations
33:17 The role of frameworks in corporations
34:32 Advantages of large teams in corporations
40:01 Challenges of technical debt in startups
43:12 Career advice for junior data scientists
44:10 Tools and frameworks for MLOps projects
49:00 Balancing new and old technologies in skill development
55:43 Data engineering challenges and reliability in LLMs
57:09 On-premise vs. cloud solutions in data-sensitive industries
59:29 Alternatives like Dask for distributed systems
```

### Step 2: Content Organization
Break down the transcript into separate text files or data elements based on the thematic chunks.

Final result is {`chunk_1`, `chunk_2`, ..., `chunk_n`} - a set of files or data elements which would contain a specific part of the transcript specified in a relevant time range.

### Step 3: Format Selection
Choose from the available carousel formats (detailed below).

### Step 4: Content Extraction
Based on the selected format, extract content using one of these approaches:

1. Key Takeaways

**Prompt:**
```
Based on the transcript, extract key takeaways. These should be the most important insights, ideas, or lessons a listener would remember from this section. Avoid vague or generic points. Use clear, specific, high-impact phrasing.
Output as a numbered list. Each item should be ≤25 words.
```

2. Notable Quotes

**Prompt:**
```
Extract memorable or insightful quotes and their context directly from the transcript. Prefer short, self-contained, high-impact quotes that could stand alone.
```

3. Guest Journey

**Prompt:**
```
Summarize the guest's career or learning journey as described in this transcript.
Break it into chronological stages using short headings + 1–2 sentence descriptions per stage.
Focus on turning points, obstacles, motivations, and transitions.
```

4. Practical Advice

**Prompt:**
```
Extract actionable advice or tips mentioned by the guest.
These should be practical, specific, and useful — not vague or motivational.
Format as a list, with each item:
- Tip in bold (max 10 words)
- Supporting sentence or example (optional, max 20 words)
```

5. Resources Mentioned

**Prompt:**
```
List all specific tools, frameworks, books, libraries, datasets, or products mentioned in the transcript.
For each, provide:
- Name of the resource
- One-sentence description or how it was used/recommended by the guest
```

### Step 5: Content Consolidation
Combine results from each chunk into a single final file, `file_final`.

So for each carousel format you would have its own `file_final`.

### Step 6: Carousel Generation
Transform the processed content from `file_final` into a markdown table used to construct LinkedIn carousel:

```
Create a carousel table from the provided content:

Structure:
1. Cover (Slide 1)
   - Title: Main topic/value (8-10 words)
   - Text: Key insight or hook (120 chars)

2. Content
   - Title: Clear point per slide
   - Text: Supporting detail or example
   - Keep related points together
   - Maximum 8 content slides

3. CTA (Final Slide)
   - Title: Action-focused
   - Text: Clear next step

Requirements:
- All titles: 8-10 words max
- All text: 120 characters max
- Professional tone
- No repetition between title and text
- Each slide should be self-contained

Table Format:
| Slide # | Title | Text |
|---------|-------|------|
| 1 | Main Topic | Hook |
| 2-9 | Point | Detail |
| 10 | CTA | Action |
```


## Carousel Format Templates

### 1. Takeaways Format
- Slide 1: "5 Key Takeaways from [Guest] on [Topic]"
- Core Slides: One takeaway per slide with quote and explanation
- Final slide: Call-to-action

### 2. Quotes Format
- Slide 1: "Thought-Provoking Quotes from [Guest Name]"
- Core Slides: Single quote per slide with minimal context
- Final slide: "Get the full context – link in comments"

### 3. Guest Journey Format
- Slide 1: "How [Guest Name] Went from X to Y"
- Core Slides: One journey stage per slide.
- Final slide: Journey-related CTA or quote

### 4. Practical Advice Format
- Slide 1: "5 Practical Tips from [Guest] on [Theme]"
- Core Slides: One tip per slide
- Final slide: Podcast CTA

### 5. Resources Format
- Slide 1: "[Guest]'s Toolkit from the Episode"
- Core Slides: One resource per slide with brief comment
- Final slide: "More recommendations in the episode!"

### Special: Topic Deep Dive Format
- Slide 1: "Let's talk about [Topic]"
- Slide 2: Topic importance
- Core Slides: Simple concept explanations
- Final slide: "Want more like this? Listen here."

## Technical Specifications

### Dimensions
- Square: 1080 x 1080 pixels
- Portrait: 1080 x 1350 pixels

### Typography Guidelines

| Text Element    | Font Size (px) |
|----------------|----------------|
| Main Headline  | 60-80         |
| Subheadline    | 40-60         |
| Body Text      | 30-45         |
| Caption        | 20-30         |

### Character Count Guidelines

| Element     | Recommended Length                    |
|------------|--------------------------------------|
| Headline   | 3-10 words (~40-60 characters)       |
| Subheading | 80-100 characters maximum            |
| Body Text  | ~120 characters (1-2 short sentences)|
| CTA/Footer | Under 50 characters                  |

> **Best Practice:** Use concise, impactful sentences. Split longer content across multiple slides.

## Style Guidelines for Your Prompts

Include into your prompts:

```
- Be factual and thorough
- Use natural language for native speakers
- Maintain professional tone
- Focus on clarity and impact
```