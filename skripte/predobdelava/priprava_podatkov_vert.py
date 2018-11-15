import re
from random import shuffle
import random


MAXLEN_SENTENCE = 70
MAXLEN_WORD = 20

K_FOLDS = 10
FOLD_WIDTH = 2762
TOTAL_SAMPLES = 27624

DATA_DIR = './podatki/'

random.seed(13)


def get_data(in_vert, maxlen_sentence, maxlen_word):
    stavki = []
    with open(in_vert) as f:
        vrstice = []
        m_beseda = 0
        for line in f:
            line_items = re.split(r'\t+', line)
            if len(line_items) == 14:
                vrstice.append({
                    'beseda': line_items[0],
                    'lema': (line_items[1].split('-')[0]
                             if line_items[2] != 'Z' else '-'),
                    'oznaka': line_items[2]})
                if len(vrstice[-1]['beseda']) > m_beseda:
                    m_beseda = len(vrstice[-1]['beseda'])
            elif line.startswith('</s>'):
                if m_beseda <= maxlen_word and len(vrstice) <= maxlen_sentence:
                    stavki.append(vrstice)
                vrstice = []
                m_beseda = 0
    return stavki


def write_sents(out_file, sents):
    with open(out_file, 'w') as outfile:
        for s in sents:
            for token in s:
                outfile.write('{}\t{}\t{}\n'.format(
                    token['beseda'],
                    token['lema'],
                    token['oznaka']))
            outfile.write('\n')


def split_k_folds(in_vert, fold_size, k_folds, maxlen_sentence,
                  maxlen_word):
    stavki = get_data(in_vert, maxlen_sentence, maxlen_word)
    shuffle(stavki)
    folds = []
    for i in range(k_folds-1):
        folds.append(stavki[i * fold_size:(i+1) * fold_size])
    folds.append(stavki[(k_folds-1)*fold_size:])

    for i in range(k_folds):
        write_sents(
            DATA_DIR+'fold{}_test.vert'.format(i+1),
            folds[i])
        train = [stavek for j, fold in enumerate(folds)
                 for stavek in fold if j != i]
        write_sents(
            DATA_DIR+'fold{}_train.vert'.format(i+1),
            train)


def main():
    """
    split_k_folds(DATA_DIR+'ssj500k20.vert',
                  FOLD_WIDTH, K_FOLDS,
                  MAXLEN_SENTENCE, MAXLEN_WORD)
    """
    split_k_folds(DATA_DIR+'ssj500k20.vert', 0, 1, MAXLEN_SENTENCE,
                  MAXLEN_WORD)



if __name__ == '__main__':
    main()
