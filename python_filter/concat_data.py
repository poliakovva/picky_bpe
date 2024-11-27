import os


def is_utf8_encoded(file_path):
    """Check if the file contains only UTF-8 characters."""
    try:
        with open(file_path, 'rb') as f:
            f.read().decode('utf-8')
        return True
    except (UnicodeDecodeError, FileNotFoundError):
        return False


def filter_txt_files(directory):
    """Filter .txt files in the given directory that contain only UTF-8 characters."""
    utf8_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            if is_utf8_encoded(file_path):
                utf8_files.append(file_path)

    return utf8_files


def concatenate_txt_files(file_paths, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # Read the content and write it to the output file
                    content = infile.read()
                    outfile.write(content + '\n')  # Add newline for separation
            except FileNotFoundError:
                print(f"Warning: The file {file_path} was not found and will be skipped.")
            except IOError as e:
                print(f"An error occurred while reading {file_path}: {e}")


if __name__ == "__main__":
    directory = "../datasets/D184MB"
    txt_files = filter_txt_files(directory)

    # Output file path
    output_file_path = f'../datasets/combined_{directory[len(txt_files)-6:]}.txt'

    # Concatenate the files
    concatenate_txt_files(txt_files, output_file_path)

    print(f"Successfully concatenated files to {output_file_path}")