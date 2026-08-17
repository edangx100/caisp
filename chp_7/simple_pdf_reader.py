# simple-pdf-reader.py
# To run this script, use `python simple-pdf-reader.py <file_path> <max_characters>`

import sys
from PyPDF2 import PdfReader

def read_pdf(file_path, max_chars):
    file_path = file_path.strip().replace("\n", "").replace("\r", "")
    # Open the PDF file and read it
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            text_content = ""
            # Extract text from each page of the PDF
            for page in reader.pages:
                text_content += page.extract_text()

        # Print the required amount of text based on max_chars
        print(text_content[:max_chars])
        return text_content[:max_chars]

    except Exception as e:
        print(f"Error reading the PDF file: {e}")

if __name__ == '__main__':
    # Take file path and max characters as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python simple-pdf-reader.py <file_path> <max_characters>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        max_chars = int(sys.argv[2])
    except ValueError:
        print("Please enter a valid integer for max_characters.")
        sys.exit(1)

    read_pdf(file_path, max_chars)
