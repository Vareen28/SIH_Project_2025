import pandas as pd
import os

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Step 1: Load and Process the CSV File ---
try:
    df = pd.read_csv('CentralReport1756823684918.csv', skiprows=7)
    df_needed = df[['S.No', 'STATE', 'DISTRICT', 'Rainfall (mm)', 'Total Geographical Area (ha)', 'Ground Water Recharge (ham)']].copy()
    df_needed.columns = ['s_no', 'state', 'district', 'rainfall_mm', 'total_geo_area_ha', 'gw_recharge_ham']
    df_needed.dropna(subset=['district'], inplace=True)

    # --- Step 2: Create Text "Documents" for RAG ---
    def create_text_representation(row):
        return (
            f"For the district of {row['district']} in the state of {row['state']}, "
            f"the annual rainfall was {row.get('rainfall_mm', 0)} mm, "
            f"the total geographical area is {row.get('total_geo_area_ha', 0)} hectares, and "
            f"the estimated annual Ground Water Recharge is {row.get('gw_recharge_ham', 0)} hectare-meters (ham)."
        )
    documents = df_needed.apply(create_text_representation, axis=1).tolist()
    
    print(f"Successfully created {len(documents)} documents.")
    if documents:
        print("Example Document:\n", documents[0])
    else:
        print("No documents were created.")
        exit()

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Step 3: Create Embeddings and the Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = FAISS.from_texts(documents, embeddings)

# --- Step 4: Set up the Local LLM ---
llm = LlamaCpp(
    model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=2048,
    verbose=False,
)
print("Model loaded successfully!")

# --- Step 5: Set up the QA Chain ---
prompt_template = """
Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# --- Step 6: Ask a Question! ---
question = "What is the groundwater recharge in the Bilaspur district?"
response = qa_chain.invoke(question)

print("\n--- Query ---")
print(question)
print("\n--- RAG Answer ---")
print(response['result'])