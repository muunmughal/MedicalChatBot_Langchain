# MedicalChatBot_Langchain
Medical chat bot with langchain , streamlit 

--------------------------------------------------------------

Step#1 : Create memory.py for Loading Document of type pdf 
Step#2 : Create Chunkes 
Step#3 : Create Vector Embedding with sentence-transformer model
Step#4 : Store Eembedding in FAISS 

---------------------------------------------------------------
---------------------------------------------------------------

Step#1 : Setup LLM with HuggingFace model
         ( provide -> model_id , temperature, hf_token, max-length )
Step#2 : Create Custom Prompt Using  Template
Step#3 : Connect LLM with FAISS Database
Step#4 : Create Question/Answer Chain 

--------------------------------------------------------------
--------------------------------------------------------------

Steamlit For User Interface Design 

--------------------------------------------------------------

**Packeges :-**


pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
-------------------------------------------
pipenv install huggingface_hub
------------------------------------------
pipenv install streamlit
-----------------------------------------



        



