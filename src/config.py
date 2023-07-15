from pathlib import Path
from dataclasses import dataclass

@dataclass
class config:

    playlist_url = 'https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI'
    DEBUG = True

    # Paths
    root_artifact_dir = Path('Downloaded_artifacts')

    # wandb
    project_name = 'Youtube_playlist_QA'
    transcription_artifacts_path = 'ghatgetanuj/Youtube_playlist_QA/Transcription_Artifacts:latest'
