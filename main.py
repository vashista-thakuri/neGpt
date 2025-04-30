from Data.cleaning import NepaliTextCleaner
from Model.modelTraining import ModelTrainer

if __name__ == "__main__":

    # Data Prep
    parent_folder = "C:/Users/vashista.thakuri/Downloads/archive/nepali_news_dataset_20_categories_large/nepali_news_dataset_20_categories_large"

    ntc = NepaliTextCleaner(parent_folder)
    clean_data = ntc.process_and_clean()

    print("Cleaned text is ready for further processing.")

    # Model Training
    trainer = ModelTrainer(clean_data)
    trainer.train()

    # Generate text using the trained model
    generated_text = trainer.generate_text()
    print(generated_text)