import os
import chardet
import re

# Initialize text variable to store preprocessed text
text = ""

# Define the NepaliTextCleaner class to handle text cleaning
class NepaliTextCleaner:
    def __init__(self, parent_folder):
        self.parent_folder = parent_folder
        self.text = text

    # Detect the encoding of a file using chardet
    def detect_encoding(self, file_path):
        """Detect the encoding of a file using first 10k characters."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result['encoding']

    # Combine all text files in the parent directory into a single text file
    def combine_files(self):
        """Combine all text files in the parent folder into a single output file."""
        combined_text = []
        for root, _, files in os.walk(self.parent_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    encoding = self.detect_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as infile:
                        combined_text.append(infile.read())
        self.text = "\n".join(combined_text)        

    # Clean the combined text to keep only Devanagari characters, spaces, and newlines
    def clean_data(self):
        """Keep only Devanagari characters, spaces, and newlines."""
        self.text = re.sub(r'[^\u0900-\u097F\s\n]', '', self.text)

        print(f"Total number of characters: {len(self.text)}")
        unique_chars = sorted(set(self.text))
        print(f"Unique characters: {len(unique_chars)}")
        print(f"Character List: {unique_chars}")

    # Return the cleaned text
    def process_and_clean(self):
        """Combine files and clean the output, returning the cleaned text."""
        self.combine_files()
        self.clean_data()
        return self.text