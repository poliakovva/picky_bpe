from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

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
            else:
                print('NOT-UTF-8', file_path)

    return utf8_files



def main():
    files_path = '../datasets/D670MB'
    file_paths = filter_txt_files(files_path)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(vocab_size=30000,
                         min_frequency=2,
                         show_progress=True,
                         special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(file_paths, trainer)

if __name__ == "__main__":
    main()
