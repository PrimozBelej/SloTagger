import neuralmodel
import tag
import teiutils
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('output_h5', type=str,
                        help='Path to output file containing model weights.')
    parser.add_argument('output_json', type=str,
                        help='Path to output file containing model '
                        'configuration.')
    parser.add_argument('-f', '--force', help='Overwrite output files.',
                        action='store_true')
    parser.add_argument('-b', '--beginning', type=int, default=0,
                        help='Index of the first sentence in the training set '
                        '(inclusive). Omitted parameter means training set '
                        'begins with the first sentence.')
    parser.add_argument('-e', '--end', type=int, default=-1,
                        help='Index of the last sentence in the training set '
                        '(exclusive). Omitted parameter means training set '
                        'ends with the last sentence.')
    return parser.parse_args()


def validate_args(args):
    if not args.input.endswith('xml'):
        print('Invalid input file extension. Expected xml.')
        exit()

    if not os.path.exists(args.input):
        print('File {} does not exist.'.format(args.input))
        exit()

    if os.path.exists(args.output_h5) and not args.force:
        print('Output file {} already exist. '
              'Output file can be overwriten by passing '
              '-f argument.'.format(args.output_h5))
        exit()

    if os.path.exists(args.output_json) and not args.force:
        print('Output file {} already exist. '
              'Output file can be overwriten by passing '
              '-f argument.'.format(args.output_json))
        exit()

    if args.beginning > args.end:
        print('Invalid range specified. End must be greater than beginning.')
        exit()


def main():
    args = parse_args()
    validate_args(args)

    sentences = list(teiutils.read(args.input, False, args.beginning, args.end))
    tags = list(teiutils.read(args.input, True, args.beginning, args.end))

    character_dict = tag.load_character_dict('./characterlist')
    x = tag.vectorize_sentences(sentences, character_dict)

    tag_dict = tag.load_tag_dict('./pos_embeddings')
    y = tag.vectorize_tags(tags, tag_dict)
    model = neuralmodel.build_model(x[0], len(character_dict), y.shape[2])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=1)
    neuralmodel.save_model(model, args.output_json)
    model.save_weights(args.output_h5)


if __name__ == '__main__':
    main()
