# =========================
# RAG (Retrieval-Augmented Generation) Example with Google GenAI and ChromaDB by krishr.bhattarai
# to-dos: make a dashboard/ UI to make the user experience better, make users able to upload documents themselves
# like pdfs and the system be able to embedded them as well. make this scalable and practical. make a readme file and push to github. MAKE A PROJECT REPOORT FOR THIS PROJECT.
# =========================

import logging
import os
# from urllib import response  # lets us work with environment variables, now removed because we switched to the modern Google GenAI SDK which has a different way of handling responses
from dotenv import load_dotenv  # loads .env file
import chromadb  # vector database
from google import genai # import the Google Generative AI library modern sdk
import logging

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
    
# Set up logging
logging.basicConfig(filename="rag_log.txt", level=logging.INFO, 
                    format="%(asctime)s - %(message)s")


# =========================
# FUNCTION: Split Text into Chunks
# =========================

def chunk_text(text, min_chunk_size=200, max_chunk_size=500):
    """
    Breaks large text into chunks based on natural breaks (like paragraphs) but keeps them within a size range.
    """
    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n\n"):  # Split by double newline (natural break)
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    # Append the last chunk if it's large enough
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
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

    # semantic embedding
    question_embedding = embed_text(question)

    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    # Keyword embedding (simple keyword extraction)
    keywords = " ".join(question.split()[:5])
    keyword_embedding = embed_text(keywords)

    keyword_results = collection.query(
        query_embeddings=[keyword_embedding],
        n_results=5
    )

    # Combine IDs with simple scoring
    combined_scores = {}

    # Add vector results
    for doc_id, distance in zip(vector_results["ids"][0],
                                 vector_results["distances"][0]):
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - distance)

    # Add keyword results
    for doc_id, distance in zip(keyword_results["ids"][0],
                                 keyword_results["distances"][0]):
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - distance)

    # Sort by combined score
    ranked_ids = sorted(combined_scores,
                        key=combined_scores.get,
                        reverse=True)

    # Take top 3
    top_ids = ranked_ids[:3]

    # Build context
    combined_context = ""
    for doc_id in top_ids:
        doc = collection.get(ids=[doc_id])
        combined_context += doc["documents"][0] + "\n\n"

     # Calculate top similarity score from results
    top_score = combined_scores[top_ids[0]]  # assuming lower is better (distance metric)
    
     # Combine top results into context
    context = combined_context  # Use the combined context instead of just the vector results
    
    # Define a confidence threshold (you can tweak this)
    confidence_threshold = 0.3  # Adjusted based on testing as correct question typically has a score around 0.3-0.4, while incorrect ones are often below 0.2  
    confidence_score = round(top_score, 3)

    if confidence_score < confidence_threshold:
        confidence_message = "Confidence too low. Please ask a more precise question."
        logging.info(f"Question: {question} | Confidence: {top_score} | Response: Low confidence")
        print(confidence_message)
        
   

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
    answer = response.text
    # Log the interaction
    logging.info(f"Question: {question} | Confidence: {top_score} | Answer: {answer}")
    
    return answer



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
         # Get user feedback
        rating = input("Did I sound smart enough? (1 = yes, 0 = no): ")
        
        # Log with feedback
        confidence_score = float(answer.split("Confidence: ")[-1].split()[0]) if "Confidence:" in answer else "N/A"  #answer.split("Confidence: ")[-1] splits the answer at the confidence: part and takes the last part(signified by -1) which is the score
        #--and stores it in confidence_score, or it will split when there is a space and take and store the first part, i.e 0, and if confidence is not in the answer it will store N/A 
        logging.info(f"Question: {question} | Answer: {answer} | Confidence: {confidence_score} | Rating: {rating}")