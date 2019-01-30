import argparse
import os
import itertools
from sklearn.metrics import accuracy_score
import teiutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('true', type=str,
                        help='Path to xml/tei file containing true tags.')
    parser.add_argument('predicted', type=str,
                        help='Path to xml/tei file containing predicted tags.')
    return parser.parse_args()


def validate_args(args):
    if not args.true.endswith('xml') or \
       not args.predicted.endswith('.xml'):
        print('Invalid input file extension. Expected xml.')
        exit()

    if not os.path.exists(args.true):
        print('File {} does not exist.'.format(args.input))
        exit()

    if not os.path.exists(args.predicted):
        print('File {} does not exist.'.format(args.predicted))
        exit()


def main():
    args = parse_args()
    validate_args(args)

    true_tags = list(itertools.chain.from_iterable(
        teiutils.read(args.true, True)))[:2000]

    predicted_tags = list(itertools.chain.from_iterable(
        teiutils.read(args.predicted, True)))[:2000]

    accuracy_pos = accuracy_score(true_tags, predicted_tags)
    accuracy_category = accuracy_score([tag[0] for tag in true_tags],
                                       [tag[0] for tag in predicted_tags])

    print('Accuracy score for POS: {}'.format(accuracy_pos))
    print('Accuracy score for category: {}'.format(accuracy_category))


if __name__ == '__main__':
    main()
