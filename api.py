import pandas as pd
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. SETUP: Load data and model ---
print("--- Initializing Chatbot API ---")
try:
    df = pd.read_csv('CentralReport1756823684918.csv', skiprows=7)
    df_needed = df[['S.No', 'STATE', 'DISTRICT', 'Rainfall (mm)', 'Total Geographical Area (ha)', 'Ground Water Recharge (ham)']].copy()
    df_needed.columns = ['s_no', 'state', 'district', 'rainfall_mm', 'total_geo_area_ha', 'gw_recharge_ham']
    df_needed.dropna(subset=['district'], inplace=True)

    def create_text_representation(row):
        return (
            f"For the district of {row['district']} in the state of {row['state']}, "
            f"the annual rainfall was {row.get('rainfall_mm', 0)} mm, "
            f"the total geographical area is {row.get('total_geo_area_ha', 0)} hectares, and "
            f"the estimated annual Ground Water Recharge is {row.get('gw_recharge_ham', 0)} hectare-meters (ham)."
        )
    documents = df_needed.apply(create_text_representation, axis=1).tolist()
    print(f"Successfully created {len(documents)} documents.")
except Exception as e:
    print(f"FATAL ERROR during data loading: {e}")
    exit()

# --- Create Embeddings and the Vector Store ---
print("Creating embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vector_store = FAISS.from_texts(documents, embeddings)
print("Vector store created successfully.")

# --- Set up the Local LLM ---
print("Loading the local language model...")
llm = LlamaCpp(
    model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    temperature=0.1,
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=2048,
    verbose=False,
)
print("Model loaded successfully!")

# --- Set up the QA Chain with a Stricter Prompt ---
# **THE CORRECTION: This prompt is more direct and leaves less room for creative answers.**
prompt_template = """
You are a data retrieval assistant. Your task is to answer questions based ONLY on the provided context.
Provide only the direct answer to the question without any introductory phrases or extra information.
If the information is not in the context, respond with 'Information not available.'

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# --- 2. API: Create the Flask Web Server ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/ask", methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        print(f"Received question: {question}") 
        response = qa_chain.invoke(question)
        answer = response['result']
        # **THE CORRECTION: A more robust way to clean the final answer.**
        # This removes any lingering parts of the prompt.
        final_answer = answer.split("Answer:")[-1].strip()
        print(f"Answer returned: {final_answer}") 
        return jsonify({"answer": final_answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 3. RUN: Start the server ---
if __name__ == '__main__':
    print("\n--- Chatbot API is ready and running! ---")
    app.run(host='0.0.0.0', port=5000, debug=False)