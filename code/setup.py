"""
this file archives the commands to build the project, 
but is't needed later
"""

raise PermissionError("This file is not needed anymore")

# %conda install conda-forge::transformers
# %python -m pip install huggingface_hub

# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")

from huggingface_hub import snapshot_download
import tqdm

snapshot_download(
    repo_id="monologg/bert-base-cased-goemotions-original", 
    local_dir="../models/bert-base-cased-goemotions-original",
    local_dir_use_symlinks=False
    )