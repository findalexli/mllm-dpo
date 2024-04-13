import os
from huggingface_hub import HfApi, Repository

# Define your Hugging Face API client
api = HfApi()

# User credentials, replace YOUR_HF_API_TOKEN with your actual Hugging Face API token
hf_username = "alexshengzhili"
hf_api_token = "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"  # Make sure to use your own API token here

# Declare hyperparameters
dpo_beta_values = [0.1, 0.3, 0.5]
learning_rates = ["5e-5", "5e-6"]
num_train_epochs_integers = [2, 3]

# Function to ensure repository exists or create it
def ensure_repository(hf_username, model_name, token):
    repo_url = f"{hf_username}/{model_name}"
    try:
        # Try to create the repo, if it doesn't exist this will succeed, else it will raise an error
        api.create_repo(repo_id=model_name, token=token, exist_ok=True)  # Adjusted line
        print(f"Repository {repo_url} created successfully.")
    except Exception as e:
        print(f"Repository {repo_url} already exists or another error: {e}")

model_name = "llava-v1.5-13b-lora-1227-COH-lrv0-3230llava0-5879_interleaved.json"
model_path = f"/home/ubuntu/latest_llava/LLaVA/checkpoints/{model_name}"

# # Loop through each combination of hyperparameters
# for dpo_beta in dpo_beta_values:
#     for learning_rate in learning_rates:
#         for num_train_epoch in num_train_epochs_integers:
#             # Configure the run name
#             model_name = f"llava-lora-dpo-1227lrvtail2000_sft-self-sampled-beta-{dpo_beta}-lr-{learning_rate}-avg-False-epoch-{num_train_epoch}"
#             model_path = f"/home/ubuntu/LLaVA/checkpoints/{model_name}"
#             repo_id = f"{hf_username}/{model_name}"
#             # Ensure the repository exists or create it
#             ensure_repository(hf_username, model_name, hf_api_token)

#             # Call upload_folder to upload the local folder's content to the repository
#             try:
#                 api.upload_folder(
#                     folder_path=model_path,
#                     repo_id=repo_id,
#                     repo_type="model",  # or "dataset" or "space" depending on your use case
#                     token=hf_api_token
#                 )
#                 print(f"Contents of {model_path} uploaded to repository {repo_id} successfully.")
#             except Exception as e:
#                 print(f"An error occurred: {e}")

