# simple-file-reader.py
# To run this script, use `python simple-file-reader.py <file_path> <max_characters>`

import sys

def read_file(file_path, max_chars):
    try:
        file_path = file_path.strip().replace("\n", "").replace("\r", "")
        # Open the file and read its content
        with open(file_path, 'r') as file:
            content = file.read()

        # Print the required amount of text based on max_chars
        print(content[:max_chars])
        return content[:max_chars]

    except Exception as e:
        print(f"Error reading the file: {e}")

if __name__ == '__main__':
    # Take file path and max characters as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python simple-file-reader.py <file_path> <max_characters>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        max_chars = int(sys.argv[2])
    except ValueError:
        print("Please enter a valid integer for max_characters.")
        sys.exit(1)

    read_file(file_path, max_chars)
