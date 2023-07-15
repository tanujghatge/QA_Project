from pytube import YouTube
from pytube import Playlist
import os
from gradio_client import Client
# import sys
# sys.path('C:\\Drive data\\projectss\\QA_PROJECT')
from config import config
import wandb
from dotenv import load_dotenv
import pandas as pd

def GetPlaylistLinks(playlist_link):
  p = Playlist(playlist_link)
  playlist_urls = list()
  for url in p.video_urls:
    playlist_urls.append(url)
  playlist_title = p.title
  return playlist_title,playlist_urls



def get_transcription(playlist_link, episodes = None):
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    playlist_title,links = GetPlaylistLinks(playlist_link)

    links = links[:10]
    # wandb
    run = wandb.init(project= config.project_name, job_type= 'upload Transcripts')
    transcription_artifacts = wandb.Artifact('Transcription_Artifacts',type = 'dataset')
    transcription_table = wandb.Table(columns = ['Title','Video_Title' ,'Video_Link','Transcription', 'Video_Length', 'Video_Data'])

    transcriptions = []
    for _ in links:
        
        # get individual video details
        yt = YouTube(_)
        video_title = yt.title
        video_length = yt.length
        video_publish_data = str(yt.publish_date)
        result = client.predict(
                        _,	# str  in 'YouTube URL' Textbox component
                        "transcribe",	# str  in 'Task' Radio component
                        False,	# bool  in 'Return timestamps' Checkbox component
                        api_name="/predict_2"
        )
        transcriptions.append(result[1])
        
        # Add row to wandb table
        transcription_table.add_data(playlist_title,video_title,_,result[1],video_length,video_publish_data)
    # Add whole table to wandb artifacts
    transcription_artifacts.add(transcription_table, 'transcription_Data')

    # Add csv file in artifacts
    csv_file_path = os.path.join(config.root_artifact_dir,"Transcription_Artifacts.csv")
    df = pd.DataFrame({"title": playlist_title,"links" : links, "transcription" : transcriptions})
    df['title'] = playlist_title
    df.to_csv(csv_file_path, index= False)
    transcription_artifacts.add_file(csv_file_path)

    # push all the files in weights and biases and end the run
    run.log_artifact(transcription_artifacts)
    run.finish()




if __name__ == "__main__":
  load_dotenv()
  wandb_api = os.environ['WANDB_API']
   
  get_transcription(config.playlist_url)
   
   






