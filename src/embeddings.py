from config import config
import wandb
import pandas as pd
import os
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter


def get_data(transcription_artifact_name : str , total_episodes = None):
    if config.DEBUG == True:
        total_episodes = 3
    run = wandb.init(project= config.project_name, job_type = 'perform_embeddings')

    # Get data from wandb artifacts
    embeddings_artifacts = wandb.use_artifact(transcription_artifact_name, type = 'dataset')
    embeddings_artifacts_dir = embeddings_artifacts.download(config.download_path)
    filename = transcription_artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(embeddings_artifacts_dir, f"{filename}.csv"))
    if total_episodes is not None:
        df = df.iloc[:total_episodes]

    run.finish()
    return df

def perform_embeddings(df: pd.DataFrame):
    loader = DataFrameLoader(df, page_content_column="transcription")
    data = loader.load()
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    return docs



if __name__ == "__main__": 
    df = get_data(config.transcription_artifacts_path)
    docs = perform_embeddings(df)
    print(docs)