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


def main():
    directory = input("./D670MB")
    utf8_files = filter_txt_files(directory)

    if utf8_files:
        print("The following .txt files contain only UTF-8 characters:")
        for file in utf8_files:
            print(file)
    else:
        print("No .txt files with only UTF-8 characters found.")


if __name__ == "__main__":
    main()