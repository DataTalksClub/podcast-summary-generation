import requests
from bs4 import BeautifulSoup

def get_transcript_from_url(episode_url):
    response = requests.get(episode_url)
    soup = BeautifulSoup(response.text, "html.parser")

    start_collecting = False
    transcript_lines = []

    for line in soup.get_text(separator="\n").splitlines():
        line = line.strip()

        if "Transcript" in line and not start_collecting:
            start_collecting = True
            continue

        if start_collecting and line:
            transcript_lines.append(line)

    return "\n".join(transcript_lines).strip()


# Fetch main podcast page
main_page_url = "https://datatalks.club/podcast.html"
main_response = requests.get(main_page_url)
main_soup = BeautifulSoup(main_response.text, "html.parser")

# Find the second latest episode link
episode_links = []
for li in main_soup.find_all("li"):
    a_tag = li.find("a")
    if a_tag and a_tag.get("href", "").startswith("/podcast/"):
        full_link = "https://datatalks.club" + a_tag["href"]
        episode_links.append(full_link)
        if len(episode_links) == 2:
            break

if len(episode_links) < 2:
    print("⚠️ Could not find the second latest episode.")
else:
    second_latest_url = episode_links[1]
    print(f"🎯 Fetching transcript from: {second_latest_url}")
    transcript = get_transcript_from_url(second_latest_url)

    if transcript:
        filename = "second_latest_transcript.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Transcript\n\n")
            f.write(f"**Source:** [{second_latest_url}]({second_latest_url})\n\n")
            f.write("---\n\n")
            f.write(transcript)
        print(f"✅ Markdown transcript saved to '{filename}'")
    else:
        print("⚠️ No valid transcript found.")
