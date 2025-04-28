import ollama

# Step 1: Load the text file
#with open("first_latest_episode_transcript.txt", "r", encoding="utf-8") as file:
#with open("second_latest_episode_transcript.txt", "r", encoding="utf-8") as file:
with open("third_latest_episode_transcript.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

# Step 2: Define summary style
summary_style = "detailed"  # Options: "short", "medium", "detailed"

# Step 3: Create precise prompt
if summary_style == "short":
    prompt = (
        "Summarize the main topics and technical content discussed in the following podcast transcript. "
        "Do NOT mention the host, guest, or any personal names. "
        "Focus only on the ideas, concepts, and main discussion points.\n\n"
        f"{input_text}"
    )
elif summary_style == "medium":
    prompt = (
        "Write a 5-7 sentence paragraph summarizing the main technical themes discussed in the podcast transcript. "
        "Do NOT refer to the speakers or authors. "
        "Focus purely on the subjects, tools, methods, and key insights covered.\n\n"
        f"{input_text}"
    )
elif summary_style == "detailed":
    prompt = (
        "Create a structured bullet-point summary of the main ideas, technical concepts, and topics discussed in the podcast transcript. "
        "Exclude all references to the speakers, hosts, or personal mentions. "
        "Focus entirely on the content discussed.\n\n"
        f"{input_text}"
    )
else:
    prompt = f"Summarize the main ideas from the following text:\n\n{input_text}"

# Step 4: Generate summary using local Phi model
response = ollama.generate(model="phi", prompt=prompt)
summary = response["response"]

# Step 5: Save result to file
with open("phi_summary_alexey_third.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("✅ Clean technical summary saved to 'phi_summary_alexey_third.txt'")

