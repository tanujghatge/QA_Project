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
  playlist_name = p.title
  return playlist_name, playlist_urls

playlist_name,links = GetPlaylistLinks('https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI')
for i,link in enumerate(links, total = 2):
  print(i,link)
    #     result = client.predict(
    #                     _,	# str  in 'YouTube URL' Textbox component
    #                     "transcribe",	# str  in 'Task' Radio component
    #                     False,	# bool  in 'Return timestamps' Checkbox component
    #                     api_name="/predict_2"
    #     )

    #     transcriptions.append(result[1])
    #     transcription_table.add_data(_,result[1])
    # transcription_artifacts.add(transcription_table, 'transcription_Data')