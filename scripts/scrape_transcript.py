import re
import requests
import argparse
import time
import os
from bs4 import BeautifulSoup


def fetch_html(url, retries=3, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; podcast-scraper/1.0)"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise HTTPError for 4xx or 5xx
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt + 1}] Error fetching {url}: {e}")
            time.sleep(delay)
    raise Exception(f"Failed to fetch {url} after {retries} attempts")




def parse_transcript(html):
    soup = BeautifulSoup(html, 'html.parser')

    results = [] 

    current_section = None

    for tag in soup.select(".content-main > *"):
        if tag.name == "h3":
            current_section = tag.get_text(strip=True)
            results.append({"type":"section", "title": current_section})

        elif tag.name == "p":
            bold = tag.find("b")
            if bold and tag.text.strip().startswith(bold.text):
                speaker = bold.text.strip()
                full_text = tag.get_text(separator=" ", strip=True)
                spoken_text = re.sub(rf"^{re.escape(speaker)}\s*:\s*", "", full_text)
                results.append({"type": "line", "speaker": speaker, "text": spoken_text})

    return results, soup

def title_to_slug(title):
    return re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')


def save_transcript_md(results, soup, output_dir="../data/transcripts"):
    os.makedirs(output_dir, exist_ok=True)

    title = soup.find("h1").get_text(strip=True)
    slug = title_to_slug(title)
    filename = os.path.join(output_dir, f"{slug}-transcript.md")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for item in results:
            if item["type"] == "section":
                f.write(f"\n## {item['title']}\n\n")
            else:
                f.write(f"**{item['speaker']}**: {item['text']}\n\n")

    print(f"Transcript saved to: {filename}")
    return filename

def scrape_and_save(url):
    html = fetch_html(url)
    results, soup = parse_transcript(html)
    return save_transcript_md(results, soup)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse and save a DataTalks Club podcast transcript.")
    parser.add_argument("--url", required=True, help="URL to the podcast episode page (HTML)")
    args = parser.parse_args()

    try:
        scrape_and_save(args.url)
    except Exception as e:
        print(f"Couldn't scrape and save, error: {e}")











