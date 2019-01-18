import argparse
from sklearn.metrics import accuracy_score
import tag
import neuralmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input file.")
    args = parser.parse_args()

    sentences, tags = tag.read_vert_file(args.input, fields=(0, 2))

    model = neuralmodel.load_model(
        './model_5fold.json',
        './model_10_1.h5'
    )
    sentences = sentences[:2000]
    tags = tags[:2000]
    predictions = tag.predict_tags(sentences, model)
    print(accuracy_score(
        [tag for sentence_tags in tags for tag in sentence_tags],
        [prediction for sentence_predictions in predictions for prediction in sentence_predictions]))


if __name__ == '__main__':
    main()
