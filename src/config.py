from pathlib import Path
from dataclasses import dataclass

@dataclass
class config:

    playlist_url = 'https://youtube.com/playlist?list=PLjq9mRS1PfGAuL41Q1vIPlSzQuuJibbhg'

    # Paths
    root_artifact_dir = Path('Downloaded_artifacts')

    # wandb
    project_name = 'Youtube_playlist_QA'
    transcription_artifacts_path = 'ghatgetanuj/Youtube_playlist_QA/Transcription_Artifacts:latest'
