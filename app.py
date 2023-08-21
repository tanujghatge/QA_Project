from src.config import config
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

import wandb

from dotenv import load_dotenv
import os
import faiss

def get_embeddings(artifact_name : str):    
    transcript_embeddings_artifacts = wandb.use_artifact(artifact_name,"dataset")
    transcript_embeddings_artifacts_dir =  transcript_embeddings_artifacts.download(config.root_artifact_dir)
    # artifact = wandb.use_artifact('ghatgetanuj/Youtube_playlist_QA/Embeddings_Artifacts:v0', type='dataset')
    # artifact_dir = artifact.download(config.root_artifact_dir)
    
# wandb.init(project=config.project_name, job_type="app")

# transcript_embeddings_artifact = wandb.use_artifact(config.transcription_embeddings_artifact, "dataset")
# transcript_embeddings_artifact.download(config.root_artifact_dir / "FAISS")
# print(faiss_db_dir)




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
    # index = faiss.read_index(r"C:\Drive data\projectss\QA_PROJECT\Downloaded_artifacts\YT_QA_EMB.faiss")
    # faiss_db = FAISS.load_local(r"C:\Drive data\projectss\QA_PROJECT\Downloaded_artifacts", embeddings, index_name="YT_QA_EMB")
    faiss_db = FAISS.load_local(config.root_artifact_dir, embeddings, index_name="YT_QA_EMB")

    # retriever = faiss_db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=faiss_db.as_retriever(),
        chain_type_kwargs={"k" : 2, "prompt": prompt},
        return_source_documents=True,
    )

    with get_openai_callback() as cb:
        result = qa({"query" : question})
        print(cb)

    answer = result["result"]
    return answer



if __name__ == "__main__":
    load_dotenv()
    wandb.init(project=config.project_name, job_type="app")
    get_embeddings(config.transcription_embeddings_artifact)
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']