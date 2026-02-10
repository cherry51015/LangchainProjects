import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langserve import add_routes
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
model=ChatGroq(model='llama-3.1-8b-instant',groq_api_key=GROQ_API_KEY)

generic_template="translate the follwoning into this {language}"
prompt=ChatPromptTemplate([("system",generic_template),('user','{input}')])

parser=StrOutputParser()
chain=prompt|model|parser

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

add_routes(app,chain,path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)

