import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the main podcast page
main_page_url = "https://datatalks.club/podcast.html"
main_response = requests.get(main_page_url)
main_soup = BeautifulSoup(main_response.text, "html.parser")

# Step 2: Find the first bullet point link
first_li_link = None
for li in main_soup.find_all("li"):
    a_tag = li.find("a")
    if a_tag and a_tag.get("href", "").startswith("/podcast/"):
        first_li_link = a_tag["href"]
        break

if first_li_link is None:
    print("⚠️ Could not find the latest episode link.")
    exit()

# Step 3: Build the full URL
episode_url = "https://datatalks.club" + first_li_link
print(f"🎯 Found latest episode link: {episode_url}")

# Step 4: Fetch the episode page
episode_response = requests.get(episode_url)
episode_soup = BeautifulSoup(episode_response.text, "html.parser")

# Step 5: Extract transcript starting from "Transcript" title
start_collecting = False
transcript_lines = []

for line in episode_soup.get_text(separator="\n").splitlines():
    line = line.strip()

    if "Transcript" in line and not start_collecting:
        start_collecting = True
        continue

    if start_collecting:
        # No filters: keep all lines
        if line:
            transcript_lines.append(line)

# Step 6: Save the transcript
transcript_text = "\n".join(transcript_lines).strip()

if transcript_text:
    with open("latest_tracscript.txt", "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"✅ Transcript from {episode_url} saved into 'latest_tracscript.txt'")
else:
    print("⚠️ No valid transcript found.")
