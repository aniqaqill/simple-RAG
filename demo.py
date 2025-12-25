import ollama
import numpy as np
import subprocess

# --- 1. WSL-TO-WINDOWS CONNECTION LOGIC ---
def get_windows_host_ip():
    """Automatically finds the Windows host IP from inside WSL."""
    try:
        # Grabs the gateway IP (your Windows side)
        cmd = "ip route | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    except Exception:
        return "127.0.0.1" # Fallback

win_ip = get_windows_host_ip()
print(f"Connecting to Windows Ollama at: http://{win_ip}:11434")

# Initialize the custom client
client = ollama.Client(host=f"http://{win_ip}:11434")

# --- 2. CONFIGURATION ---
# Note: Ensure you have run 'ollama pull <model_name>' on Windows first!
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# --- 3. DATA LOADING ---
dataset = []
try:
    with open('cat-facts.txt', 'r') as file:
        dataset = [line.strip() for line in file.readlines() if line.strip()]
        print(f'Loaded {len(dataset)} cat facts.')
except FileNotFoundError:
    print("Error: cat-facts.txt not found. Please create it in the same folder.")
    dataset = ["Cats spend 70% of their lives sleeping.", "A group of cats is called a clowder."]

# --- 4. VECTOR DATABASE SETUP ---
VECTOR_DB = []

def add_chunk_to_database(chunk):
    # Use 'client.embed' instead of 'ollama.embed'
    response = client.embed(model=EMBEDDING_MODEL, input=chunk)
    embedding = response['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

print("Indexing documents into Vector DB...")
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    if (i + 1) % 5 == 0: print(f'Added {i+1}/{len(dataset)} chunks...')

# --- 5. RAG FUNCTIONS ---
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_n=3):
    # Use 'client.embed' for the query
    query_embedding = client.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- 6. CHATBOT EXECUTION ---
input_query = input('\nAsk a question about cats: ')
retrieved_knowledge = retrieve(input_query)

print('\n--- Retrieved Knowledge ---')
for chunk, similarity in retrieved_knowledge:
    print(f' - (Sim: {similarity:.2f}) {chunk}')

# Build the context-aware prompt
context_text = '\n'.join([f' - {chunk}' for chunk, sim in retrieved_knowledge])
instruction_prompt = f'''You are a helpful chatbot. 
Use only the following context to answer. If the answer isn't here, say you don't know.
Context:
{context_text}'''

print('\n--- Chatbot Response ---')
stream = client.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print("\n")