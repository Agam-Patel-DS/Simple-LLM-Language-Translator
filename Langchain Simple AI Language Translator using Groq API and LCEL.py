
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt Template
generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Output Parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# Streamlit app
st.title("Langchain with Streamlit")

language = st.text_input("Enter the language:")
text = st.text_area("Enter the text:")

if st.button("Translate"):
    if language and text:
        result = chain.invoke({"language": language, "text": text})
        st.write(result)
    else:
        st.warning("Please enter both language and text.")
