from huggingface_hub import hf_hub_download

repo_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"

model_path = hf_hub_download(
    repo_id=repo_id,
    filename="nemotron-3.5-asr-streaming-0.6b.nemo",
    local_dir="./models",
)

print("Model downloaded:")
print(model_path)