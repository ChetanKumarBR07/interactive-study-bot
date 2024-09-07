import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
import streamlit as st

# Ensure your GOOGLE_API_KEY is set in the environment
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
st.title('Welcome to StudyBot')

genai.configure(api_key=GOOGLE_API_KEY)

# Load documents
documents = SimpleDirectoryReader(input_files=["IEEE.pdf"]).load_data(show_progress=True)

# Define embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Define our LLM. In this case, we choose to use Google's Gemini
Settings.llm = Gemini(model="models/gemini-1.5-flash")
model = genai.GenerativeModel('gemini-1.5-flash')
# Create the vector store to query data from
index = VectorStoreIndex.from_documents(
    documents,
)
# Create the vector store to query data from
query_engine = index.as_query_engine(streaming=True)


def generate_response(query):
    """Generates a response to a user query using Google's Gemini LLM."""
    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")  # Print the error message
        return "Sorry, I encountered an error. Please try again later."  # Return a user-friendly message

def docread():
  while True:
   user_query = input("You: ")
   if user_query.lower() == "exit":
    print("Going to General Chat Mode....")
    break
   else:
    streaming_response = query_engine.query(user_query)
    streaming_response.print_response_stream()
   

if __name__ == "__main__":
    print("Hello! I am a StudyBot. How can I assist you today?")
    while True:
       weight = st.number_input("Enter your weight (in kgs)")
       user_query = input("You: ")
       if user_query.lower() == "go to document read mode":
        print("Going to Document Read Mode....")
        docread()
       elif user_query.lower() == "exit":
        print("Shutting Down")
        break
       else:
         chatbot_response = generate_response(user_query)
         print("StudyBot:", chatbot_response)
       
