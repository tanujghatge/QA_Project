from config import config
import wandb
import pandas as pd
import os
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone, FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import pinecone
from dotenv import load_dotenv


def get_data(transcription_artifact_name : str , total_episodes = None):
    if config.DEBUG == True:
        total_episodes = 3
    # run = wandb.init(project= config.project_name, job_type = 'get_data')

    # Get data from wandb artifacts
    embeddings_artifacts = wandb.use_artifact(transcription_artifact_name, type = 'dataset')
    embeddings_artifacts_dir = embeddings_artifacts.download(config.root_artifact_dir)
    filename = transcription_artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(embeddings_artifacts_dir, f"{filename}.csv"))
    if total_episodes is not None:
        df = df.iloc[:total_episodes]

    # run.finish()
    return df

def perform_embeddings(df: pd.DataFrame, OPENAI_API_KEY):
    # Load data in form of langchain
    loader = DataFrameLoader(df, page_content_column="transcription")
    data = loader.load()

    # SPlit data using tokens
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    
    print("total number of docuements are: ", len(docs))
    # Get embeddings for data splitted using chunks

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Pinecone Integration
    # pinecone.init(api_key=pinecone_api_key,
    #               environment='asia-southeast1-gcp-free')
    
    # # docs_search = Pinecone.from_texts([d.page_content for d in docs], embeddings, index_name=config.pinecone_index_name)
    # docs_search = Pinecone.from_documents(docs, embeddings, index_name=config.pinecone_index_name)


    # ChromaDB

    # FAISS
    FAISS_DB = FAISS.from_documents(docs, embedding=embeddings)
    FAISS_DB.save_local(folder_path=config.root_artifact_dir,index_name='YT_QA_EMB')

    # Wandb
    # run = wandb.init(project= config.project_name, job_type = 'perform_embeddings')
    embeddings_artifacts = wandb.Artifact('Embeddings_Artifacts',type = 'dataset')
    file_to_upload = os.path.join(config.root_artifact_dir,'YT_QA_EMB.faiss')
    embeddings_artifacts.add_file(file_to_upload)
    wandb.log_artifact(embeddings_artifacts)
    # run.finish()   
    
    return FAISS_DB

def get_conversation_chain(vectorspace):
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                                               retriever=vectorspace.as_retriever(),
                                                               memory = memory)
    return conversation_chain



if __name__ == "__main__": 
    wandb.init(project=config.project_name, job_type="Embeddings")
    load_dotenv()

    openai_api = os.environ['OPENAI_API_KEY']
    pinecone_api = os.environ['PINECONE_API_KEY']

    df = get_data(config.transcription_artifacts_path)
    docs = perform_embeddings(df,openai_api)
    print(type(docs))
    query = "What are transformers?"
    ans = docs.similarity_search(query)
    print(ans[0].page_content)
    wandb.finish()