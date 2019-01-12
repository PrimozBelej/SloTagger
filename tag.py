import numpy as np
import neuralmodel
import poslib
import argparse
import os


def read_vert_file(filename, fields=(0, 2)):
    contents = open(filename).readlines()

    result = [[] for f in fields]
    sentence_values = [[] for f in fields]

    sents = []
    sent = []

    tags = []
    sent_tags = []

    for row_index, row in enumerate(contents):
        if len(row.strip()) == 0:
            for i, sv in enumerate(sentence_values):
                result[i].append(sv)
            sentence_values = [[] for f in fields]
            continue

        split_row = row.strip().split('\t')
        if len(split_row) < max(fields)+1:
            raise IndexError("Unexpected number of columns in row {} of file "
                             "{}.".format(row_index, filename))
        for field_index, field in enumerate(fields):
            sentence_values[field_index].append(split_row[field].strip())

    if sentence_values[0]:
        for i, sv in enumerate(sentence_values):
            result[i].append(sv)
    return result


def filter_sents(sents, tags, maxlen_word, maxlen_sent):
    filtered_sents = []
    filtered_tags = []
    for i, sent in enumerate(sents):
        if len(sent) <= maxlen_sent:
            filtered_sents.append(sent)
            filtered_tags.append(tags[i])
    sents = filtered_sents
    tags = filtered_tags

    filtered_sents = []
    filtered_tags = []
    for i, sent in enumerate(sents):
        if max([len(word) for word in sent]) <= maxlen_word:
            filtered_sents.append(sent)
            filtered_tags.append(tags[i])
    sents = filtered_sents
    tags = filtered_tags
    return sents, tags


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


def sent2embedding(sent, maxlen_sent, maxlen_word, character_dict):
    embedding = np.zeros((maxlen_sent, maxlen_word))
    for i, word in enumerate(sent):
        for j, char in enumerate(word):
            embedding[i, j] = character_dict[char]
    return embedding


def vectorize_tags(tags, maxlen_sent, tag_dict):
    tag_dim = len(list(tag_dict.values())[0])
    y = np.zeros((len(tags), maxlen_sent, tag_dim))
    for i, tag in enumerate(tags):
        y[i] = sent_tags2embedding(tag, maxlen_sent, tag_dict, tag_dim)
    return y


def vectorize_sentences(sentences, maxlen_sent, maxlen_word, character_dict):
    x = np.zeros((len(sentences), maxlen_sent, maxlen_word))
    for i, sentence in enumerate(sentences):
        x[i] = sent2embedding(
            sentence, maxlen_sent, maxlen_word, character_dict)
    return x


def sent_tags2embedding(oznake_stavka, maxlen_sent, tag_dict, tag_dim):
    embedding = np.zeros((maxlen_sent, tag_dim))
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
        sent_tags = mostprobablepos(embeddings, pos_index, prediction[i, :])
        if local_pred2tag:
            binary = binarize(prediction[i, :], list(poslib.PROPERTY_INDEX.values()))
            str_emb = ''.join(list(binary.astype('str')))
            if str_emb in embedding_dict:
                sent_tags = embedding_dict.get(str_emb)
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


def write_vert(path, sentences, tags):
    assert len(sentences) == len(tags)
    with open(path, 'w') as outfile:
        for sentence_index, sentence in enumerate(sentences):
            sentence_tags = tags[sentence_index]
            assert len(sentence_tags) == len(sentence)
            for word_index, word in enumerate(sentence):
                outfile.write(word)
                outfile.write('\t')
                outfile.write(sentence_tags[word_index])
                outfile.write('\n')
            outfile.write('\n')



def predict_tags(sentences, model):
    maxlen_word = 20
    maxlen_sent = 70
    character_dict = load_character_dict('./characterlist')
    character_dim = len(character_dict)
    tag_dict = load_tag_dict('./pos_embeddings')
    tag_dim = len(list(tag_dict.values())[0])
    x = vectorize_sentences(sentences, maxlen_sent, maxlen_word, character_dict)
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


def get_sentences_txt(file_path):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input file.")
    parser.add_argument("output", help="Path to output file.")
    parser.add_argument("-f", "--force", help="Overwrite output file.",
                        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File {} does not exist.".format(args.input))
        exit()

    if os.path.exists(args.output) and not args.force:
        print("Output file {} already exist. "
              "Output file can be overwriten by passing "
              "-f argument.".format(args.output))
        exit()

    if args.input.endswith(".vert"):
        sentences = read_vert_file(args.input, fields=(0,))[0]
    elif args.input.endswith(".txt"):
        sentences = get_sentences_txt(args.input)
    else:
        print("Invalid input file extension. "
              "Valid input types are vert and txt.")
        exit()

    model = neuralmodel.load_model(
        './model_5fold.json',
        #'./model_10_1.h5'
        './model_celotni_podatki.h5'
    )
    #sents, tags = filter_sents(sents, tags, maxlen_word, maxlen_sent)

    #predictions = predict_tags([['Moj', 'pes', 'je', 'zelo', 'lep', '.']], model)
    sentences = sentences[:3]
    predictions = predict_tags(sentences[:3], model)
    write_vert(args.output, sentences, predictions)

    # Omejitve dolzin besed in povedi

    # Podatke preberemo iz datoteke vert
    #sents, tags = read_vert_file('./podatki/fold1_test.vert')

    # Odstranimo predolge povedi
    #model = neuralmodel.build_model(x[0], character_dim, tag_dim)

    #x, y = load_data(sents, tags, maxlen_word, maxlen_sent, character_dict,
    #                 tag_dict)

    # Zgradimo model
    """
    model = build_model(x[0, :, :], maxlen_word, maxlen_sent, character_dim,
                        tag_dim)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    """

    # Naucimo model
    """
    model.fit(x[1:], y[1:], epochs=30)
    """


    # Napovemo oznake testne povedi


if __name__ == '__main__':
    main()
