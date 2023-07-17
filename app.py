from src.config import config
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

import wandb

from dotenv import load_dotenv
import os

# def get_embeddings(artifact_name : str):
    
#     transcript_embeddings_artifacts = wandb.use_artifact(artifact_name,"dataset")
#     transcript_embeddings_artifacts_dir =  transcript_embeddings_artifacts.download(config.root_artifact_dir)

    
wandb.init(project=config.project_name, job_type="app")

transcript_embeddings_artifact = wandb.use_artifact(config.transcription_embeddings_artifact, "dataset")
transcript_embeddings_artifact.download(config.root_artifact_dir / "FAISS")
# print(faiss_db_dir)


load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def get_answer(question : str):
    prompt_template = """Use the following pieces of context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't add your opinions or interpretations. Ensure that you complete the answer.
    If the question is not relevant to the context, just say that it is not relevant.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:"""

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    faiss_db = FAISS.load_local(r"C:\Drive data\projectss\QA_PROJECT\Downloaded_artifacts\YT_QA_EMB.faiss" , embeddings)
    return faiss_db
    # retriever = faiss_db.as_retriever()
    # retriever.search_kwargs["k"] = 2

    # qa = RetrievalQA.from_chain_type(
    #     llm=ChatOpenAI(temperature=0),
    #     chain_type="stuff",
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": prompt},
    #     return_source_documents=True,
    # )


db = get_answer("Hello")
print(db)
