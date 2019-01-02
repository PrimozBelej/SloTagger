# -*- coding: utf-8 -*-
import re
import argparse


MAXLEN_SENTENCE = 80
MAXLEN_WORD = 20

DATA_DIR = './podatki/'


def get_data(in_vert, maxlen_sentence, maxlen_word):
    with open(in_vert) as in_file:
        lines = []
        m_beseda = 0
        for line in in_file:
            line_items = re.split(r'\t+', line)
            if len(line_items) == 14:
                lines.append((
                    line_items[0],
                    (line_items[1].split('-')[0]
                     if line_items[2] != 'Z' else '-'),
                    line_items[2]))
                if len(lines[-1][0]) > m_beseda:
                    m_beseda = len(lines[-1][0])
            elif line.startswith('</s>'):
                if m_beseda <= maxlen_word and len(lines) <= maxlen_sentence:
                    yield lines
                lines = []
                m_beseda = 0


def preprocess(in_vert, out_tsv, maxlen_sentence, maxlen_word):
    sentences = get_data(in_vert, maxlen_sentence, maxlen_word)
    with open(out_tsv, 'w') as outfile:
        for sentence in sentences:
            for token in sentence:
                outfile.write('{}\t{}\t{}\n'.format(
                    token[0],
                    token[1],
                    token[2]))
            outfile.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='Path to input file (*.vert).')
    parser.add_argument('output', type=str,
                        help='Path to output file (*.tsv).')

    args = parser.parse_args()
    preprocess(args.input, args.output, MAXLEN_SENTENCE, MAXLEN_WORD)


if __name__ == '__main__':
    main()
