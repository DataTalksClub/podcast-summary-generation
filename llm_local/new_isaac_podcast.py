from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# Initialize the Ollama Phi model locally
llm = Ollama(model="phi" , temperature=0.5)

# Load the txt file 
#txt_path = 'first_latest_episode_transcript.txt'
txt_path = 'second_latest_episode_transcript.txt'
#txt_path = 'third_latest_episode_transcript.txt'
#with open(txt_path, 'r') as f:
    #txt = f.read()

# Try utf-8-sig if UTF-8 gives issues
with open(txt_path, "r", encoding="utf-8-sig") as f:
    txt = f.read()

# Get segments from txt by splitting on periods (.)
segments = txt.split('.')

# Put the period back in each segment
segments = [segment.strip() + '.' for segment in segments if segment.strip()]

# Further split by commas
segments = [segment.split(',') for segment in segments]

# Flatten the list of segments
segments = [item.strip() for sublist in segments for item in sublist if item.strip()]

from langchain_ollama import OllamaLLM
import pandas as pd

# Initialize the Ollama Phi model locally
llm = OllamaLLM(model="phi", temperature=0.5)

def create_sentences(segments, MIN_WORDS, MAX_WORDS):
    # Combine the non-sentences together
    sentences = []
    is_new_sentence = True
    sentence_length = 0
    sentence_num = 0
    sentence_segments = []

    for i in range(len(segments)):
        if is_new_sentence == True:
            is_new_sentence = False
        # Append the segment
        sentence_segments.append(segments[i])
        segment_words = segments[i].split(' ')
        sentence_length += len(segment_words)

        # If exceed MAX_WORDS, then stop at the end of the segment
        # Only consider it a sentence if the length is at least MIN_WORDS
        if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
            sentence = ' '.join(sentence_segments)
            sentences.append({
                'sentence_num': sentence_num,
                'text': sentence,
                'sentence_length': sentence_length
            })
            # Reset
            is_new_sentence = True
            sentence_length = 0
            sentence_segments = []
            sentence_num += 1

    return sentences

def create_chunks(sentences, CHUNK_LENGTH, STRIDE):
    sentences_df = pd.DataFrame(sentences)
  
    chunks = []
    for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
        chunk = sentences_df.iloc[i:i+CHUNK_LENGTH]
        chunk_text = ' '.join(chunk['text'].tolist())
        
        chunks.append({
            'start_sentence_num': chunk['sentence_num'].iloc[0],
            'end_sentence_num': chunk['sentence_num'].iloc[-1],
            'text': chunk_text,
            'num_words': len(chunk_text.split(' '))
        })
    
    chunks_df = pd.DataFrame(chunks)
    return chunks_df.to_dict('records')

def parse_title_summary_results(results):
    out = []
    for e in results:
        e = e.replace('\n', '')
        if '|' in e:
            processed = {'title': e.split('|')[0],
                        'summary': e.split('|')[1][1:]
                        }
        elif ':' in e:
            processed = {'title': e.split(':')[0],
                        'summary': e.split(':')[1][1:]
                        }
        elif '-' in e:
            processed = {'title': e.split('-')[0],
                        'summary': e.split('-')[1][1:]
                        }
        else:
            processed = {'title': '',
                        'summary': e
                        }
        out.append(processed)
    return out
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
#from langchain.llms import OllamaLLM  # If OllamaLLM is part of LangChain's LLM module
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama  # Ensure you use the correct Ollama LLM wrapper
from datetime import datetime

def summarize_stage_1(chunks_text):
  
    print(f'Start time: {datetime.now()}')

    # Prompt to get title and summary for each chunk
    map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text:
{text}

Return your answer in the following format:
Title | Summary...
e.g.
Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

TITLE AND CONCISE SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Define the LLM (using Ollama)
    ollama_llm = Ollama(model="phi", temperature=0.5)

    # Create LLMChain with the prompt template and Ollama LLM
    map_llm_chain = LLMChain(llm=ollama_llm, prompt=map_prompt)

    # Prepare input for the chain
    map_llm_chain_input = [{'text': t} for t in chunks_text]

    # Run the chain with the input data
    map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

    # If the result is a single string, split it accordingly
    if isinstance(map_llm_chain_results, str):
        # Handle the string output and split by your delimiter (e.g., "|")
        map_llm_chain_results = [{'text': result} for result in map_llm_chain_results.splitlines()]

    # Process results
    stage_1_outputs = parse_title_summary_results([e['text'] for e in map_llm_chain_results])

    print(f'Stage 1 done time {datetime.now()}')

    return {
        'stage_1_outputs': stage_1_outputs
    }



# Define your segments and parameters (MIN_WORDS, MAX_WORDS)
segments = txt.split('.')  # or however you are splitting the text into segments
segments = [segment + '.' for segment in segments]

MIN_WORDS = 5  # Example value, adjust as needed
MAX_WORDS = 20  # Example value, adjust as needed

# Create sentences from segments
sentences = create_sentences(segments, MIN_WORDS, MAX_WORDS)

# Now create chunks using these sentences
CHUNK_LENGTH = 3  # Example value, adjust as needed
STRIDE = 1  # Example value, adjust as needed

chunks_text = [chunk['text'] for chunk in create_chunks(sentences, CHUNK_LENGTH, STRIDE)]

# Run the summarization stage
stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']

# Split the titles and summaries
stage_1_summaries = [e['summary'] for e in stage_1_outputs]
stage_1_titles = [e['title'] for e in stage_1_outputs]

# Count the number of chunks
num_1_chunks = len(stage_1_summaries)


from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Initialize local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed summaries and titles
summary_embeds = np.array(embedding_model.embed_documents(stage_1_summaries))
title_embeds = np.array(embedding_model.embed_documents(stage_1_titles))


from scipy.spatial.distance import cosine
import numpy as np

import numpy as np
from scipy.spatial.distance import cosine

# Initialize similarity matrix with zeros
summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))

# Fill the matrix with cosine similarity scores
for row in range(num_1_chunks):
    for col in range(row, num_1_chunks):
        # Ensure the embeddings don't contain NaN or Inf before calculating similarity
        if np.any(np.isnan(summary_embeds[row])) or np.any(np.isnan(summary_embeds[col])):
            similarity = np.nan  # Set to NaN if any embedding is invalid
        else:
            similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])

        # Assign the computed similarity to both [row, col] and [col, row]
        summary_similarity_matrix[row, col] = similarity
        summary_similarity_matrix[col, row] = similarity  # Symmetric

# If there are NaN values in the matrix, handle them (e.g., replace with 0 or another value)
summary_similarity_matrix = np.nan_to_num(summary_similarity_matrix, nan=0.0)

# Draw a heatmap with the summary_similarity_matrix
plt.figure()
# Color scheme blues
plt.imshow(summary_similarity_matrix, cmap = 'Blues')
plt.show()

import numpy as np
import networkx as nx
from networkx.algorithms import community

def get_topics(title_similarity, num_topics=8, bonus_constant=0.25, min_size=3):
    """
    Apply Louvain community detection with proximity bonus to find topics.
    """

    # Apply proximity bonus to similarity matrix
    proximity_bonus_arr = np.zeros_like(title_similarity)
    for row in range(proximity_bonus_arr.shape[0]):
        for col in range(proximity_bonus_arr.shape[1]):
            if row != col:
                proximity_bonus_arr[row, col] = (1 / abs(row - col)) * bonus_constant

    title_similarity += proximity_bonus_arr

    # Convert to graph
    title_nx_graph = nx.from_numpy_array(title_similarity)

    desired_num_topics = num_topics
    resolution = 0.85
    resolution_step = 0.01
    iterations = 40

    # Initial search for resolution to get desired number of topics
    topics_title = []
    while len(topics_title) not in {desired_num_topics, desired_num_topics + 1, desired_num_topics + 2}:
        topics_title = community.louvain_communities(
            title_nx_graph, weight='weight', resolution=resolution
        )
        resolution += resolution_step

    # Prepare to find best clustering
    topics_title_accepted = []
    lowest_sd_iteration = 0
    lowest_sd = float('inf')

    for i in range(iterations):
        candidate_topics = community.louvain_communities(
            title_nx_graph, weight='weight', resolution=resolution
        )
        topic_sizes = [len(c) for c in candidate_topics]
        sizes_sd = np.std(topic_sizes)

        topics_title_accepted.append(candidate_topics)

        if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
            lowest_sd = sizes_sd
            lowest_sd_iteration = i

    # Select best clustering based on SD of topic sizes
    topics_title = topics_title_accepted[lowest_sd_iteration]
    print(f'Best SD: {lowest_sd:.4f}, Best iteration: {lowest_sd_iteration}')

    # Convert each topic set to a list and sort topics by average chunk index to make order interpretable
    topic_id_means = [np.mean(list(c)) for c in topics_title]  # Convert to list here
    topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key=lambda x: x[0])]

    # Assign each chunk to a topic
    chunk_topics = [None] * title_similarity.shape[0]
    for topic_id, chunk_indices in enumerate(topics_title):
        for idx in chunk_indices:
            chunk_topics[idx] = topic_id

    return {
        'chunk_topics': chunk_topics,
        'topics': topics_title
    }


# Calculate number of topics: 1/4 of total chunks or 8 (whichever is smaller)
num_topics = min(int(num_1_chunks / 4), 8)

# Detect topics using similarity matrix
topics_out = get_topics(
    title_similarity=summary_similarity_matrix,
    num_topics=num_topics,
    bonus_constant=0.2
)

chunk_topics = topics_out['chunk_topics']
topics = topics_out['topics']

# Plot the topic assignments for each chunk
plt.figure(figsize=(10, 4))
plt.imshow(np.array(chunk_topics).reshape(1, -1), aspect='auto', cmap='tab20')
plt.show()
# Draw vertical lines to separate chunks
for i in range(1, len(chunk_topics)):
    plt.axvline(x=i - 0.5, color='black', linewidth=0.5)

plt.yticks([])  # Hide y-axis ticks
plt.xticks(range(len(chunk_topics)), fontsize=8)
plt.title("Chunk Topic Assignments")
plt.tight_layout()
plt.show()

import ollama
from datetime import datetime

import ollama
from datetime import datetime
def summarize_stage_2(stage_1_outputs, topics, summary_num_words=300):
    from datetime import datetime
    print(f'Stage 2 start time {datetime.now()}')

    # Prompt templates
    title_prompt_template = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible,
and are different from each other:
{text}

Return your answer in a numbered list, with new line separating each title:
1. Title 1
2. Title 2
3. Title 3

TITLES:
"""

    map_prompt_template = """Write a 75-100 word summary of the following text:
{text}

CONCISE SUMMARY:"""

    combine_prompt_template = f'Write a {summary_num_words}-word summary of the following, removing irrelevant information. Finish your answer:\n{{text}}\n\n{summary_num_words}-WORD SUMMARY:'

    # Prepare data
    topics_data = []
    for c in topics:
        topic_data = {
            'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
            'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
        }
        topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
        topic_data['titles_concat'] = ', '.join(topic_data['titles'])
        topics_data.append(topic_data)

    topics_summary_concat = [c['summaries_concat'] for c in topics_data]
    topics_titles_concat = [c['titles_concat'] for c in topics_data]

    # Construct titles input
    topics_titles_concat_all = ''
    for i, c in enumerate(topics_titles_concat):
        topics_titles_concat_all += f'{i+1}. {c}\n'

    # Title generation via Ollama
    title_llm_input = title_prompt_template.format(text=topics_titles_concat_all)
    title_llm_result = ollama.chat(model="phi", messages=[{"role": "user", "content": title_llm_input}])

    # Extract title content
    try:
        titles_raw = title_llm_result.get('message', {}).get('content', '')
        titles = [t.strip() for t in titles_raw.split('\n') if t.strip()]
    except Exception as e:
        print(f"⚠️ Failed to extract titles: {e}")
        titles = []

    # Map stage
    summaries = []
    for doc in topics_summary_concat:
        map_input = map_prompt_template.format(text=doc)
        try:
            map_result = ollama.chat(model="phi", messages=[{"role": "user", "content": map_input}])
            map_summary = map_result.get('message', {}).get('content', '')
        except Exception as e:
            print(f"⚠️ Failed during map stage: {e}")
            map_summary = ''
        summaries.append(map_summary)

    # Reduce stage
    final_summary = []
    for summary in summaries:
        combine_input = combine_prompt_template.format(text=summary)
        try:
            combine_result = ollama.chat(model="phi", messages=[{"role": "user", "content": combine_input}])
            combined = combine_result.get('message', {}).get('content', '')
        except Exception as e:
            print(f"⚠️ Failed during reduce stage: {e}")
            combined = ''
        final_summary.append(combined)

    # Output
    stage_2_outputs = [{'title': t, 'summary': s} for t, s in zip(titles, final_summary)]

    out = {
        'stage_2_outputs': stage_2_outputs,
        'final_summary': ' '.join(final_summary)
    }

    print(f'Stage 2 done time {datetime.now()}')
    return out


# Query GPT-3 to get a summarized title for each topic_data
out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
stage_2_outputs = out['stage_2_outputs']
stage_2_titles = [e['title'] for e in stage_2_outputs]
stage_2_summaries = [e['summary'] for e in stage_2_outputs]
final_summary = out['final_summary']

print(stage_2_outputs)

print(final_summary)


#output_file = "first_transcript_summary_isaac.txt"
output_file = "second_transcript_summary_isaac.txt"
#output_file = "third_transcript_summary_isaac.txt"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Final Summary:\n\n")
        f.write(final_summary.strip())  # Ensure clean formatting
    print(f"✅ Summary successfully saved to: {output_file}")
except Exception as e:
    print(f"❌ Error saving summary: {e}")

