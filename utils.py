from pytube import YouTube
from pytube import Playlist
import os


def playlist_to_videos(playlist_link):
  p = Playlist(playlist_link)
  playlist_urls = list()
  for url in p.video_urls:
    playlist_urls.append(url)
  playlist_name = p.title
  return playlist_urls, playlist_name