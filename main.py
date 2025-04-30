from Data.cleaning import NepaliTextCleaner

if __name__ == "__main__":
    parent_folder = "/path/to/your/parent/directory"  # Replace with your actual path

    ntc = NepaliTextCleaner(parent_folder)
    clean_data = ntc.process_and_clean()

    print("Cleaned text is ready for further processing.")