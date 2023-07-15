from config import config
import wandb
import pandas as pd
import os
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import pinecone
from dotenv import load_dotenv


def get_data(transcription_artifact_name : str , total_episodes = None):
    if config.DEBUG == True:
        total_episodes = 3
    run = wandb.init(project= config.project_name, job_type = 'perform_embeddings')

    # Get data from wandb artifacts
    embeddings_artifacts = wandb.use_artifact(transcription_artifact_name, type = 'dataset')
    embeddings_artifacts_dir = embeddings_artifacts.download(config.root_artifact_dir)
    filename = transcription_artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(embeddings_artifacts_dir, f"{filename}.csv"))
    if total_episodes is not None:
        df = df.iloc[:total_episodes]

    run.finish()
    return df

def perform_embeddings(df: pd.DataFrame, OPENAI_API_KEY, pinecone_api_key):
    # Load data in form of langchain
    loader = DataFrameLoader(df, page_content_column="transcription")
    data = loader.load()

    # SPlit data using tokens
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)\
    
    # Get embeddings for data splitted using chunks

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    pinecone.init(api_key=pinecone_api_key,
                  environment='asia-southeast1-gcp-free')
    
    docs_search = Pinecone.from_texts([d.page_content for d in docs], embeddings, index_name=config.pinecone_index_name)
    return docs_search

def get_conversation_chain(vectorspace):
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                                               retriever=vectorspace.as_retriever(),
                                                               memory = memory)
    return conversation_chain



if __name__ == "__main__": 
    load_dotenv()

    openai_api = os.environ['OPENAI_API_KEY']
    pinecone_api = os.environ['PINECONE_API_KEY']
    df = get_data(config.transcription_artifacts_path)
    docs = perform_embeddings(df,openai_api, pinecone_api)
    