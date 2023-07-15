from config import config
import wandb
import pandas as pd
import os

def get_data(transcription_artifact_name : str , total_episodes = None):
    run = wandb.init(project= config.project_name, job_type = 'perform_embeddings')
    embeddings_artifacts = wandb.use_artifact(transcription_artifact_name, type = 'dataset')
    embeddings_artifacts_dir = embeddings_artifacts.download(config.download_path)
    filename = transcription_artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(embeddings_artifacts_dir, f"{filename}.csv"))
    run.finish()
    return df

def perform_embeddings(dataframe : pd.DataFrame):
    pass


run = wandb.init(project= config.project_name, job_type= 'upload Transcripts')
podcast_artifact = wandb.use_artifact(config.transcription_artifacts_path, type="dataset")
podcast_artifact_dir = podcast_artifact.download(config.root_artifact_dir)
filename = config.transcription_artifacts_path.split(":")[0].split("/")[-1]
df = pd.read_csv(os.path.join(podcast_artifact_dir, f"{filename}.csv"))
run.finish()
print(df)