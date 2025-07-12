import os
import re
import argparse

def parse_chunks_from_markdown(md_path):
    """
    Parses a markdown file into a list of chunks.
    Each chunk is a dict with keys: title, start, end, transcript.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"\*\*Chunk \d+: (?P<title>.+?)\*\*\n"
        r"\*\*Start - End:\*\* (?P<start>[\d:]+) - (?P<end>[\d:]+)\n"
        r"\*\*Transcript:\*\*\n(?P<transcript>.*?)(?=\n\*\*Chunk|\Z)",
        re.DOTALL
    )

    chunks = []
    for match in pattern.finditer(content):
        chunks.append({
            "title": match.group("title").strip(),
            "start": match.group("start").strip(),
            "end": match.group("end").strip(),
            "transcript": match.group("transcript").strip()
        })

    return chunks

def save_chunks_to_folder(chunks, folder="chunks"):
    """
    Saves each chunk to its own markdown file in the specified folder.
    """
    os.makedirs(folder, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):
        filepath = os.path.join(folder, f"chunk{i}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {chunk['title']}\n")
            f.write(f"**Start - End:** {chunk['start']} - {chunk['end']}\n\n")
            f.write("**Transcript:**\n\n")
            f.write(chunk['transcript'])

    print(f"✅ {len(chunks)} chunks saved to folder '{folder}'.")

def main():
    parser = argparse.ArgumentParser(description="Split a Markdown podcast file into timestamp-based chunks.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input Markdown file")
    parser.add_argument("--output_dir", "-o", default="chunks", help="Folder to save individual chunk files")
    args = parser.parse_args()

    chunks = parse_chunks_from_markdown(args.input)
    save_chunks_to_folder(chunks, args.output_dir)

if __name__ == "__main__":
    main()
