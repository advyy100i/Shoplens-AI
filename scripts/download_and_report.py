import os
import sys
from pathlib import Path
print('Starting model downloads and reporting...')

# local imports will trigger downloads
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel, pipeline
import torch

models_info = []

# 1) sentence-transformers/all-MiniLM-L6-v2
print('\nLoading sentence-transformers/all-MiniLM-L6-v2...')
st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# try to get param counts
try:
    pt = st._first_module()
    params = sum(p.numel() for p in pt.parameters())
except Exception:
    params = None
models_info.append(('sentence-transformers/all-MiniLM-L6-v2', params))

# 2) openai/clip-vit-base-patch32
print('\nLoading openai/clip-vit-base-patch32...')
proc = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
params = sum(p.numel() for p in clip.parameters())
models_info.append(('openai/clip-vit-base-patch32', params))

# 3) facebook/bart-large-cnn summarizer
print('\nLoading facebook/bart-large-cnn (this will be large)...')
summ = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if torch.cuda.is_available() else -1)
# get underlying model
try:
    bmodel = summ.model
    params = sum(p.numel() for p in bmodel.parameters())
except Exception:
    params = None
models_info.append(('facebook/bart-large-cnn', params))

# locate cache directory
home = Path.home()
hf_home = os.environ.get('HF_HOME') or os.path.join(home, '.cache', 'huggingface', 'hub')
print('\nHugging Face cache base:', hf_home)

# helper size
def dir_size(path: Path):
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

print('\nModel disk sizes and parameter counts:')
for model_id, param_count in models_info:
    # search for directories containing model_id parts under hf_home
    name = model_id.split('/')[-1]
    found_paths = []
    base = Path(hf_home)
    if base.exists():
        for p in base.rglob('*'):
            if p.is_dir() and name in p.name:
                found_paths.append(p)
    # fallback: check transformers cache folder
    if not found_paths:
        # also look for folders named model_id escaped
        for p in base.rglob('*'):
            if p.is_dir() and model_id.replace('/', '_') in p.name:
                found_paths.append(p)
    total_bytes = 0
    for p in found_paths:
        total_bytes += dir_size(p)
    # present
    mb = total_bytes / (1024*1024)
    param_info = f"{param_count:,}" if param_count else 'unknown'
    ram_est = 'unknown'
    if param_count:
        ram_est = f"~{(param_count*4)/(1024**3):.2f} GB (FP32)"
    print(f"- {model_id}: disk ~{mb:.2f} MB across {len(found_paths)} cache dirs; params={param_info}; ram_est={ram_est}")

# total cache size for these models (approx)
print('\nDone.')
