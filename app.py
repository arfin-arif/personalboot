from flask import Flask, render_template, request
import requests
import json
import re
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import RetrievalQAWithSourcesChain
load_dotenv() # load environment variables from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    try:
        #create embeddings of the PDF
        embeddings = OpenAIEmbeddings()
        persist_directory = 'db'
        vectordb = Chroma(persist_directory=persist_directory, embedding_function= embeddings) #loading pre trained vectors
    except:
        return render_template('index.html', search_result= "PLease train data to start QNA..." ,search_title = "Fetching Error")
    
    prompt_template = """Use the following pieces of context to answer the users question. Your name is "Ask PDF".
        If you don't know the answer, just say that "I don't know", don't try to make up an answer. Talk in the language in user language.
        ----------------
        {summaries}
    
        """
    #setting up prompt remplate for the GPT
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    #loading LLM so that it could answer to queries.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1200)  # Modify model_name if you have access to GPT-4
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    ans = chain(query)
    return render_template('index.html', search_result= ans['answer'] ,search_title = query)



if __name__ == '__main__':
    app.run(debug=True)
