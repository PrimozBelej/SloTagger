import argparse
import os
import subprocess
import atexit
import numpy as np
import neuralmodel
import poslib
import tsvutils
import txtutils
from config import MAXLEN_SENTENCE, MAXLEN_WORD


atexit.register(txtutils.remove_tokens_file)

tag_dict = {}
with open('en_sl_tag') as tagfile:
    for line in tagfile:
        en, sl = line.strip().split('\t')
        tag_dict[en] = sl


def load_character_dict(charindex_path):
    characters = set()
    with open(charindex_path) as charfile:
        for line in charfile:
            characters |= set(line.strip())
    characters = sorted(list(characters))
    character_dict = {char: i+1 for (i, char) in enumerate(characters)}
    return character_dict


def load_tag_dict(tagindex_path):
    tagset = open(tagindex_path).readlines()
    tagset = [o.strip().split(', ') for o in tagset]
    tag_dict = {o: np.array(list(v)) for o, v in tagset}
    return tag_dict


def sent2embedding(sent, character_dict):
    embedding = np.zeros((MAXLEN_SENTENCE, MAXLEN_WORD))
    for i, word in enumerate(sent):
        for j, char in enumerate(word):
            try:
                embedding[i, j] = character_dict[char]
            except IndexError as e:
                print(len(sent))
                print(i)
                print(j)
                exit()
    return embedding


def vectorize_tags(tags, tag_dict):
    tag_dim = len(list(tag_dict.values())[0])
    y = np.zeros((len(tags), MAXLEN_SENTENCE, tag_dim))
    for i, tag in enumerate(tags):
        y[i] = sent_tags2embedding(tag, tag_dict, tag_dim)
    return y


def vectorize_sentences(sentences, character_dict):
    x = np.zeros((len(sentences), MAXLEN_SENTENCE, MAXLEN_WORD))
    for i, sentence in enumerate(sentences):
        x[i] = sent2embedding(
            sentence, character_dict)
    return x


def sent_tags2embedding(oznake_stavka, tag_dict, tag_dim):
    embedding = np.zeros((MAXLEN_SENTENCE, tag_dim))
    for i, tag in enumerate(oznake_stavka):
        embedding[i] = tag_dict[tag]
    return embedding


def binarize(values, prop_index):
    result = np.zeros(len(values), dtype='int')
    for (begin, end) in prop_index:
        max_i = np.argmax(values[begin:end])
        result[begin+max_i] = 1
    return result


def mostprobablepos(embeddings, pos_index, prediction):
    return pos_index[np.argmax(embeddings.dot(prediction))]


def prediction2tags(prediction, embedding_dict, local_pred2tag=True):
    tags = list()
    pos_index, embeddings = posembeddings()
    for i in range(prediction.shape[0]):
        sent_tags = None
        if local_pred2tag:
            binary = binarize(prediction[i, :], list(poslib.PROPERTY_INDEX.values()))
            str_emb = ''.join(list(binary.astype('str')))
            if str_emb in embedding_dict:
                sent_tags = embedding_dict.get(str_emb)
            else:
                sent_tags = mostprobablepos(embeddings, pos_index, prediction[i, :])
        else:
            sent_tags = mostprobablepos(embeddings, pos_index, prediction[i, :])
        tags.append(sent_tags)
    return tags


def posembeddings():
    pos_index = list()
    pos_vector = np.zeros((1900, 118))
    with open('./pos_embeddings') as infile:
        for i, line in enumerate(infile):
            pos, emb = line.strip().split(', ')
            pos_vector[i, :] = np.array(list(emb)).astype('int')
            pos_index.append(pos)
    return pos_index, pos_vector


def predict_tags(sentences, model):
    character_dict = load_character_dict('./characterlist')
    x = vectorize_sentences(sentences, character_dict)
    y_predicted = model.predict(x)
    embedding_dict = {}
    predictions = []
    with open('./pos_embeddings') as infile:
        for line in infile:
            pos, embedding = line.strip().split(', ')
            embedding_dict[embedding] = pos
    for i, prediction in enumerate(y_predicted):
        predictions.append(
            prediction2tags(prediction[:len(sentences[i]), :], embedding_dict))
    return predictions


def get_sentences_txt(file_path, obeliks_path):
    pass


def eng2slo(tags):
    for tag_i, tag in enumerate(tags):
        tags[tag_i] = tag_dict[tag]
    return tags


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('output', type=str, help='Path to output file.')
    parser.add_argument('--obelikspath', type=str,
                        help='Path to Obeliks4J tokeniser '
                        'directory. Required in case of txt files.')
    parser.add_argument('-f', '--force', help='Overwrite output file.',
                        action='store_true')
    parser.add_argument('-s', '--slo', help='Return slovene tags.',
                        action='store_true')
    parser.add_argument('-bs', '--batchsize', type=int,
                        help='Number of sentences '\
                        'tagged at the same time.')
    return parser.parse_args()


def validate_args(args):
    if not os.path.exists(args.input):
        print('File {} does not exist.'.format(args.input))
        exit()

    if os.path.exists(args.output) and not args.force:
        print('Output file {} already exist. '
              'Output file can be overwriten by passing '
              '-f argument.'.format(args.output))
        exit()


def get_sentences(args):
    if args.input.endswith('.tsv'):
        sentences = tsvutils.read(args.input, fields=(0,))[0]
    elif args.input.endswith('.xml'):
        sentences = None
    elif args.input.endswith('.vert'):
        sentences = None
    elif args.input.endswith('.txt'):
        obeliks_path = args.obelikspath
        if obeliks_path is None:
            print('When using txt files as input, path to Obeliks4J tokeniser '
                  'has to be provided using --obelikspath argument.')
            exit()
        if not os.path.isdir(obeliks_path):
            print('Invalid Obeliks4J path provided: {}'.format(obeliks_path))
            exit()
        txtutils.tokenize(args.input, obeliks_path)
        sentences = txtutils.get_sentences(args.batchsize)
    else:
        print('Invalid input file extension. '
              'Valid input types are xml/tei, vert, tsv and txt.')
        exit()
    return sentences


def main():
    args = parse_args()
    validate_args(args)

    sentence_batches = get_sentences(args)

    model = neuralmodel.load_model(
        './model_5fold.json',
        './model_10_1.h5'
    )

    for batch_index, batch in enumerate(sentence_batches):
        print('Tagging batch {}.'.format(batch_index+1))
        predictions = predict_tags(batch, model)
        if args.slo:
            for tags_i, tags in enumerate(predictions):
                predictions[tags_i] = eng2slo(tags)
        print(predictions)

    #tsvutils.write(args.output, sentences, predictions)


if __name__ == '__main__':
    main()
