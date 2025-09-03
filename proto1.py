import pandas as pd
import os

# --- Imports for Local LLM and LangChain ---
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Step 1: Load and Process the CSV File ---
try:
    df = pd.read_csv('CentralReport1756823684918.csv', skiprows=7)

    df_needed = df[[
        'S.No', 'STATE', 'DISTRICT', 'Rainfall (mm)',
        'Total Geographical Area (ha)', 'Ground Water Recharge (ham)'
    ]].copy()

    df_needed.columns = [
        's_no', 'state', 'district', 'rainfall_mm',
        'total_geo_area_ha', 'gw_recharge_ham'
    ]

    df_needed.dropna(subset=['district'], inplace=True)

    # --- Step 2: Create Text "Documents" for RAG ---
    def create_text_representation(row):
        rainfall = row.get('rainfall_mm', 0)
        area = row.get('total_geo_area_ha', 0)
        recharge = row.get('gw_recharge_ham', 0)
        
        return (
            f"For the district of {row['district']} in the state of {row['state']}, "
            f"the annual rainfall was {rainfall} mm. "
            f"The total geographical area is {area} hectares, and the "
            f"estimated annual Ground Water Recharge is {recharge} hectare-meters (ham)."
        )

    documents = df_needed.apply(create_text_representation, axis=1).tolist()
    
    print(f"Successfully created {len(documents)} documents.")
    if documents:
        print("Example Document:\n", documents[0])
    else:
        print("No documents were created.")
        exit()

except FileNotFoundError:
    print("Error: Make sure 'CentralReport1756823684918.csv' is in the directory.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Step 3: Create Embeddings and the Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = FAISS.from_texts(documents, embeddings)

# --- Step 4: Set up the Local LLM ---
# This line now points to the model file you downloaded.
llm = LlamaCpp(
    model_path="Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf",
    n_gpu_layers=-1, # Offloads all layers to GPU if available
    n_batch=512,     # Should be between 1 and n_ctx, consider the amount of VRAM
    n_ctx=4096,      # Context window size
    verbose=False,   # Suppress verbose output
)

# --- Step 5: Set up the QA Chain ---
prompt_template = """
Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
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