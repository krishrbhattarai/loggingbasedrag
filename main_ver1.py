# =========================
# IMPORTS
# =========================

import os
# from urllib import response  # lets us work with environment variables, now removed because we switched to the modern Google GenAI SDK which has a different way of handling responses
from dotenv import load_dotenv  # loads .env file
import chromadb  # vector database
from google import genai # import the Google Generative AI library modern sdk


# =========================
# LOAD API KEY
# =========================

# This loads the .env file so Python can access your API key
load_dotenv()

# This creates a client object that talks to OpenAI
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# This creates a ChromaDB client (vector database in memory)
chroma_client = chromadb.Client()

# This creates a collection (like a table in a database)
collection = chroma_client.get_or_create_collection(name="docs")


# =========================
# FUNCTION: Load Text File
# =========================

def load_text(file_path):
    """
    Opens a text file and returns its full content as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# FUNCTION: Split Text into Chunks
# =========================

def chunk_text(text, chunk_size=500):
    """
    Breaks large text into smaller pieces.
    Why? Because LLMs and embeddings work better on smaller chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


# =========================
# FUNCTION: Create Embedding
# =========================

def embed_text(text):
    """
    Converts text to Gemini embedding using google.genai
    """
    result = client.models.embed_content(
        model="gemini-embedding-001", # Updated to the current standard model
        contents=text
    )
    # CHANGED: result.embeddings[0].values is the correct path for this SDK
    return result.embeddings[0].values


# =========================
# FUNCTION: Store Chunks in Vector DB
# =========================

def store_chunks(chunks):
    """
    For each chunk:
    1. Convert to embedding
    2. Store in ChromaDB
    """
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        collection.add(
            documents=[chunk],     # original text
            embeddings=[embedding],# vector version
            ids=[str(i)]           # unique ID
        )


# =========================
# FUNCTION: Ask Question
# =========================

def ask_question(question):
    """
    Steps:
    1. Convert question to embedding
    2. Find most similar text chunks
    3. Send them to LLM as context
    """

    # Convert question to vector
    question_embedding = embed_text(question)

    # Search vector DB for most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Combine top results into context
    context = "\n".join(results["documents"][0])

    # Prompt sent to LLM
    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """

    # Send prompt to gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text


# =========================
# MAIN PROGRAM
# =========================

if __name__ == "__main__":

    # Load document
    text = load_text("sample.txt")

    # Split into chunks
    chunks = chunk_text(text)

    # Store in vector database
    store_chunks(chunks)

    print("Document read!. Shoot me your questions.\n")

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAnswer:\n", answer)