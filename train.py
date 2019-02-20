import neuralmodel
import tag
import teiutils
import os
import argparse
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('output', type=str, help='Path to output directory.')
    parser.add_argument('-b', '--beginning', type=int, default=0,
                        help='Index of the first sentence in the training set '
                        '(inclusive). Omitted parameter means training set '
                        'begins with the first sentence.')
    parser.add_argument('-e', '--end', type=int, default=-1,
                        help='Index of the last sentence in the training set '
                        '(exclusive). Omitted parameter means training set '
                        'ends with the last sentence.')
    parser.add_argument('-s', '--slo', action='store_true',
                        help='Tags in input file are in slovene language.')
    parser.add_argument('-n', '--nepoch', type=int, default=20,
                        help='Number of training epoch.')
    return parser.parse_args()


def validate_args(args):
    if not args.input.endswith('xml'):
        print('Invalid input file extension. Expected xml.')
        exit()

    if not os.path.exists(args.input):
        print('File {} does not exist.'.format(args.input))
        exit()

    if not os.path.exists(args.output):
        print('Output directory {} does not exist. Creating output directory.')
        os.mkdir(args.output)

    if not os.path.isdir(args.output):
        print('Target at output path is not a directory.')
        exit()

    if args.end != -1 and args.beginning > args.end:
        print('Invalid range specified. End must be greater than beginning.')
        exit()

def get_character_set(sentences):
    character_set = set()
    for sentence in sentences:
        character_set |= set(itertools.chain.from_iterable(sentence))
    return sorted(list(character_set))


def write_character_set(charset, path):
    with open(path, 'w') as outfile:
        for char in charset:
            outfile.write(char+'\n')


def main():
    args = parse_args()
    validate_args(args)

    sentences = list(teiutils.read(args.input, False, args.beginning, args.end))
    tags = list(teiutils.read(args.input, True, args.beginning, args.end))

    charset = get_character_set(sentences)
    write_character_set(charset, args.output+'/characterlist')
    character_dict = tag.load_character_dict(args.output+'/characterlist')
    x = tag.vectorize_sentences(sentences, character_dict)

    tag_dict = tag.load_tag_dict('./pos_embeddings', args.slo)
    y = tag.vectorize_tags(tags, tag_dict)
    model = neuralmodel.build_model(x[0], len(character_dict), y.shape[2])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=args.nepoch)
    neuralmodel.save_model(model, args.output+'/model.json')
    model.save_weights(args.output+'/model_weights.h5')


if __name__ == '__main__':
    main()
