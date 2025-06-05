import os 

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatHuggingFace  # NEW
from langchain_huggingface import HuggingFaceEndpoint



# step 1: Setup  LLM ( Mistral and Hugging Face Transformers)--------------------
HF_TOKEN=os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_REPO_ID2 = "HuggingFaceH4/zephyr-7b-beta"  # CHAT-FRIENDLY MODEL


# FUNCTION 
#=================================================================



# def load_llm(huggingface_repo_id):
#     # Load the Hugging Face chat model (for conversational tasks)
#     llm = ChatHuggingFace(
#         repo_id=huggingface_repo_id,
#         huggingfacehub_api_token=HF_TOKEN,
#         model_kwargs={
#             "max_new_tokens": 512,
#             "temperature": 0.5,
#         }
#     )
#     return llm


#==================================================================

def load_llm(huggingface_repo_id):
    # Load the Hugging Face model
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        # CORRECT WAY: Pass max_new_tokens directly as a keyword argument
        max_new_tokens=512,
        temperature=0.5, # Temperature is also a direct argument
        huggingfacehub_api_token=HF_TOKEN,
        # model_kwargs should ONLY be used for parameters NOT directly exposed
        # by the HuggingFaceEndpoint constructor.
        model_kwargs={} # Empty dict if no other specific model-level kwargs are needed
    )
    return llm

#===================================================================
# def load_llm(huggingface_repo_id):
    # llm=HuggingFaceEndpoint(
    #     repo_id=huggingface_repo_id,
    #     temperature=0.5,
    #     model_kwargs={"token":HF_TOKEN,
    #                   "max_length":"512"}
    # )
    # return llm
#===========================================================


# Step 2 :  Connect LLm with FAISS and create chain 

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

#custom prompt ----------------------------------------------


def set_custom_prompt(custom_prompt_template):
    Prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return Prompt

# Load DATABASE------------------------------------
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ensure your FAISS directory exists and is populated
if not os.path.exists(DB_FAISS_PATH):
    print(f"Error: FAISS database path '{DB_FAISS_PATH}' does not exist.")
    print("Please ensure your FAISS database is created and saved before running this script.")
    exit() # Exit if DB not found

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    print("Please ensure the FAISS database is not corrupted and matches the embedding model.")
    exit()



#chain QA 

# Create QA chain ------------------------------------------------------------------------------------
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID2),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':1}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)
# ---------------------------------------------------------------------------------------------------------



# invode with single Quries

# Now invoke with a single query
user_query = input("Write Query Here: ")

try:
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
except Exception as e:
    print(f"An error occurred during QA chain invocation: {e}")
    print("This might be due to issues with the LLM, context length, or prompt.")
