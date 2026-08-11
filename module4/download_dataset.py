"""
Sources the dataset from the huggingface repository
NOTE: All filenames except data_dir are in the repo

Taken from 2 - Batch Sizes & Data Reuse.ipynb
"""
from huggingface_hub import snapshot_download
import pathlib
import shutil

path_out = pathlib.Path(__file__).parent / "data"
path_out.mkdir(exist_ok=True)

snapshot_download(
    repo_id="ChrisMcCormick/basic-arithmetic",
    repo_type="dataset",
    local_dir=str(path_out),
)

for fname in ["requirements-colab-vllm.txt"]: # in the data repo
    src = path_out / fname
    if src.exists():
        shutil.copy(src, fname)