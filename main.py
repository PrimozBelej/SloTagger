import numpy as np
import neuralmodel
import poslib


def read_vert_file(filename):
    contents = open(filename).readlines()

    sents = []
    sent = []

    tags = []
    sent_tags = []

    for row in contents:
        row = row.strip().split('\t')
        if len(row) == 3:
            sent.append(row[0])
            sent_tags.append(row[2])
        else:
            sents.append(sent)
            sent = []
            tags.append(sent_tags)
            sent_tags = []

    if sent:
        sents.append(sent)
        tags.append(sent_tags)
    return sents, tags


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


def prediction2tags(prediction, embedding_dict):
    tags = list()
    pos_index, embeddings = posembeddings()
    for i in range(prediction.shape[0]):
        binary = binarize(prediction[i, :], list(poslib.PROPERTY_INDEX.values()))
        str_emb = ''.join(list(binary.astype('str')))
        if str_emb in embedding_dict:
            tags.append(embedding_dict.get(str_emb))
        else:
            tags.append(
                mostprobablepos(embeddings, pos_index, prediction[i, :]))
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


def main():
    # Omejitve dolzin besed in povedi
    maxlen_word = 20
    maxlen_sent = 70

    # Podatke preberemo iz datoteke vert
    #sents, tags = read_vert_file('./podatki/fold1_test.vert')

    # Odstranimo predolge povedi
    #sents, tags = filter_sents(sents, tags, maxlen_word, maxlen_sent)

    # Povedi in oznake problikujemo v vektorsko obliko
    character_dict = load_character_dict('./characterlist')
    character_dim = len(character_dict)
    tag_dict = load_tag_dict('./pos_embeddings')
    tag_dim = len(list(tag_dict.values())[0])

    sentences = [['Moj', 'pes', 'je', 'zelo', 'lep', '.']]
    x = vectorize_sentences(sentences, maxlen_sent, maxlen_word, character_dict)

    model = neuralmodel.load_model(
        './model_5fold.json',
        './model_10fold_fold1.h5')
    #model = neuralmodel.build_model(x[0], character_dim, tag_dim)
    y_predicted = model.predict(x[:1])

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

    embedding_dict = {}
    with open('./pos_embeddings') as infile:
        for line in infile:
            pos, embedding = line.strip().split(', ')
            embedding_dict[embedding] = pos
    return y_predicted, embedding_dict
    #for i, prediction in enumerate(y_predicted):
    #    prediction = prediction2tags(prediction[:len(sentences[i]), :], embedding_dict)
    #    print(sentences[0])
    #    print(prediction)


    #a = y_predicted[0, :len(sents[0]), :]
    #print(a.shape)
    #prediction = prediction2tags(a,
    #                             embedding_dict)

    #print(tags[0])
    #print(prediction)


if __name__ == '__main__':
    main()
