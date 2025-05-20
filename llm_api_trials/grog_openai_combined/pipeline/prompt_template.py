def build_prompt(podcast_text):
    return f"""You are an expert in summarizing podcast episodes into engaging and informative LinkedIn carousel content designed for professionals.

Your task is to process the given podcast transcript and generate a plain text summary with the following structured elements:

1. **Title**
   - Write a compelling and informative title.
   - The title must be between 7 and 15 words.
   - It should clearly reflect the main topic of the episode.

2. **Episode Summary** 
   - Write a short summary (2–3 full sentences) giving the main idea of the episode.
   - This should include the key problem/topic, and two main highlights or takeaways.
   - Avoid using pronouns like he, she, or they; focus on ideas and insights.

3. **Guest Introduction**
   - Begin with a short introduction of the guest.
   - Write one or two complete sentences.
   - Each sentence should be between 25 and 40 words.
   - Include:
     - Full name of the guest.
     - Job title or professional role.
     - Area of expertise.
     - Reason the guest is relevant to the episode topic.

4. **Main Themes**
   - Identify three main themes discussed in the episode.
   - For each theme:
     - Write a subtitle in full-sentence form (7–12 words).
     - Under each subtitle, write four bullet points.
     - Each bullet point must be a complete sentence.
     - Each bullet point should be between 12 and 22 words.
     - Bullet points must capture meaningful and valuable professional insights.

5. **Important Software Tools**
   - Create a section titled: “Important Software Tools”
   - List important tools or platforms mentioned in the podcast.
   - For each tool:
     - Include a one-sentence explanation of what it does or how it was used.
     - Sentence must be no longer than 14 words.

6. **Guest Contact Information**

   - Create a section titled: "Guest Contact Information".
   - Search the input transcript or Markdown for a section labeled "Links:".
   - Extract all guest-related links formatted in Markdown, such as:

    * [Platform](https://link.url){{:target="_blank"}}

   - Include all valid links: LinkedIn, Twitter, GitHub, ADPList, email, or personal websites.
   - The LinkedIn profile link is **mandatory if available** and must appear **first** in this section.
   - If the LinkedIn link is not present, include the following line exactly:
    "LinkedIn link not provided in the transcript."


7. **How to Reach**
   -  Create a section titled: "How to Reach".
   - Search the transcript or Markdown for any public podcast URLs, official episode links, or contact pages.
   -   These may appear in Markdown format such as:

   * [Label](https://link.url){{:target="_blank"}}

  - Extract and include all valid URLs or links that help the listener reach the podcast.
  - Present each link on a new line or as a bulleted list using plain text or Markdown syntax.
8. **Call to Action**
   - End the summary with a call-to-action.
   - Encourage the reader to listen to the full episode.
   - Briefly explain what value they will gain by listening.

9. **Writing Style Guidelines**
   - Use a professional, polished, and energetic tone throughout.
   - Tailor the writing style to LinkedIn's professional audience.
   - Avoid:
     - Incomplete phrases.
     - Headings without context.
     - Markdown formatting.
     - Overly generic summaries.
   - Focus on:
     - Clarity.
     - Usefulness.
     - Fully written sentences.
     - Providing value to curious and career-focused professionals.

    Podcast Text:
    {podcast_text}
    """
