import requests
import yaml
import argparse
import os
from urllib.parse import urlparse


def get_podcasts_content(url: str) -> tuple[str, str]:
    '''
    This is to get the podcasts from the github repo of DataTalks.
    Link: https://github.com/DataTalksClub/datatalksclub.github.io/tree/main/_podcast 
    
    '''

    path = urlparse(url).path
    filename = os.path.basename(path)  # e.g., 's02e04-mlops.md'
    slug = os.path.splitext(filename)[0]

    response = requests.get(url)
    response.raise_for_status()

    return response.text, slug




def yaml_content_to_markdown(yaml_text: str, slug, output_dir="../data/transcripts"):
    """
    Convert YAML string content (from get_podcasts_content) into Markdown and save it.

    Parameters:
        yaml_text (str): YAML content as a string.
        md_path (str): Output Markdown file path.
    """

    filename = os.path.join(output_dir, f"{slug}-transcript.md")



    documents = list(yaml.safe_load_all(yaml_text))
    data = documents[0] 
    md_lines = []

    # Title
    md_lines.append(f"# {data.get('title', 'Untitled')}\n")

    # Guests
    guests = data.get('guests', [])
    if guests:
        md_lines.append(f"**Guests:** {', '.join(guests)}\n")

    # Transcript
    transcript = data.get('transcript', [])
    if transcript:
        md_lines.append("## Transcript\n")
        for entry in transcript:
            if 'header' in entry:
                md_lines.append(f"\n### {entry['header']}")
            elif 'line' in entry:
                who = entry.get('who', 'Unknown')
                line = entry['line'].strip()
                md_lines.append(f"**{who}**: {line}")

    markdown_output = '\n\n'.join(md_lines)

    # Print Markdown preview
    print("🔍 Markdown Preview:\n")
    print(markdown_output)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    print(f"\nMarkdown saved to: {filename}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse and save a DataTalks Club podcast transcript.")
    parser.add_argument("--url", required=True, help="URL to the podcast episode github raw content")
    args = parser.parse_args()

    yaml_text, slug= get_podcasts_content(args.url)
    yaml_content_to_markdown(yaml_text, slug)

    

