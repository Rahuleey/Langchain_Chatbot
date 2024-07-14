from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("LANGCHAIN_API_KEY")

# Debugging: Ensure API key is loaded
if api_key:
    st.write("API Key loaded successfully.")
    os.environ["LANGCHAIN_API_KEY"] = api_key
else:
    st.write("API Key not found. Please check your .env file.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title('Langchain Demo With LLAMA3 API')
input_text = st.text_input("Ask Anything")

# Debugging: Display user input
st.write(f"User input: {input_text}")

# Ollama LLaMA3 LLM
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Debugging: Check if chain is invoked correctly
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(f"Chain response: {response}")
    except Exception as e:
        st.write(f"Error invoking chain: {e}")
else:
    st.write("Please enter a question.")
