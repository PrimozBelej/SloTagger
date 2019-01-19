import subprocess
import os


TOKENIZED_FILE_PATH = 'tokens_obeliks.tsv'


def remove_tokens_file():
    if os.path.exists(TOKENIZED_FILE_PATH):
        os.remove(TOKENIZED_FILE_PATH)


def tokenize(input_path, obeliks_path):
    subprocess.run([
        'java',
        '-cp', obeliks_path+'target/classes',
        'org.obeliks.Tokenizer',
        '-if', input_path,
        '-o', TOKENIZED_FILE_PATH])


def get_sentences(batch_size):
    sentences = []
    sentence = []
    with open(TOKENIZED_FILE_PATH) as tokens:
        for line in tokens:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence.append(parts[1])
            else:
                sentences.append(sentence)
                sentence = []
                if batch_size is not None and len(sentences) == batch_size:
                    yield sentences
                    sentences = []
    if sentences:
        yield sentences

