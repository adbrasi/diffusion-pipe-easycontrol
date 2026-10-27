# usage: upload_run.py <run_name> <repo_subdir> [config_yaml]
import os, sys
from huggingface_hub import HfApi
run, sub = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"
out = f"/workspace/output/{run}"
api.upload_folder(repo_id=repo, folder_path=out, path_in_repo=sub, allow_patterns=["*.safetensors", "config.yaml", "samples/*", "*.json"])
g = f"/workspace/outputs_for_user/{run}"
if os.path.isdir(g): api.upload_folder(repo_id=repo, folder_path=g, path_in_repo=f"{sub}/grids")
if len(sys.argv) > 3: api.upload_file(repo_id=repo, path_or_fileobj=sys.argv[3], path_in_repo=f"{sub}/{os.path.basename(sys.argv[3])}")
print("UPLOADED", run, "->", sub)
