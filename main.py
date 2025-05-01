from Data.cleaning import NepaliTextCleaner
from Model.modelTraining import ModelTrainer, device
import torch

def prepare_data(parent_folder, cleaned_data_path):
    """
    Prepares and cleans the data, then writes it to a file.
    """
    clean_data = NepaliTextCleaner(parent_folder).process_and_clean()

    with open(cleaned_data_path, "w", encoding="utf-8") as f:
        f.write(clean_data)

    print("Cleaned text is ready for further processing.")

def load_model(path):
    """
    Load the model if it has already been trained.
    """
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    print(f"Entire model loaded from {path} successfully.")

    return model

def generate_text(model, decoder, max_new_tokens):
    """
    Generate text using the trained model.
    """
    context = torch.zeros((1, 1), dtype=torch.long, device=device) 
    generated_text = model.generate(context, max_new_tokens=max_new_tokens)
    decoded_text = decoder.decode(generated_text[0].tolist())

    print("Generated Text:")
    print(decoded_text)

if __name__ == "__main__":

    # Data Preparation
    parent_folder = "nepali_news_dataset_20_categories_large" # To be configured depending on the dataset location
    cleaned_data_path = "Data/cleaned_data.txt"
    prepare_data(parent_folder, cleaned_data_path)

    # Model Initialization
    neGpt = ModelTrainer()

    # Training or Loading the Model
    model_path = "Model/neGpt.pth"
    operation = "inference" # To be configured: "train" or "inference"
    
    if operation == "train":
        neGpt.train(model_path)
        print("Model trained and saved as neGpt.pth")
    else:
        model = load_model(model_path)

        # Inference
        generate_text(model, decoder=neGpt, max_new_tokens=1000)
    
    print("Process completed successfully.")