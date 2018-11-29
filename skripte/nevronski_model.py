from keras import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input, Conv1D, Concatenate,\
    GlobalMaxPool1D, Dense, Lambda, Add, Multiply, Masking, Bidirectional,\
    Dropout, TimeDistributed
import numpy as np
from poslib import embedding2tag


def read_vert_file(filename):
    contents = open(filename).readlines()

    sents = []
    sent = []

    tags = []
    sent_tags = []

    for row in contents:
        row = row.strip().split('\t')
        if len(row) == 2:
            sent.append(row[0])
            sent_tags.append(row[1])
        else:
            sents.append(sent)
            sent = []
            tags.append(sent_tags)
            sent_tags = []

    if len(sent) > 0:
        sents.append(sent)
        tags.append(sent_tags)
    return sents, tags


def filter_sents(sents, tags, maxlen_word, maxlen_sent):
    filtered_sents = []
    filtered_tags = []
    for i in range(len(sents)):
        if len(sents[i]) <= maxlen_sent:
            filtered_sents.append(sents[i])
            filtered_tags.append(tags[i])
    sents = filtered_sents
    tags = filtered_tags
    filtered_sents = []
    filtered_tags = []
    for i in range(len(sents)):
        if max([len(word) for word in sents[i]]) <= maxlen_word:
            filtered_sents.append(sents[i])
            filtered_tags.append(tags[i])
    sents = filtered_sents
    tags = filtered_tags
    return sents, tags


def load_data(sents, tags, maxlen_word, maxlen_sent):
    characters = set()
    for s in sents:
        for word in s:
            characters |= set(word)
    characters = sorted(list(characters))
    character_dict = {char: i+1 for (i, char) in enumerate(characters)}
    character_dim = len(character_dict)

    tagset = open('./podatki/oznake_vlozitve_2').readlines()
    tagset = [o.strip().split(', ') for o in tagset]
    tag_dict = {o: np.array(list(v)) for o, v in tagset}
    tag_dim = len(list(tag_dict.values())[0])

    x = np.zeros((len(sents), maxlen_sent, maxlen_word))
    y = np.zeros((len(tags), maxlen_sent, tag_dim))
    for i in range(len(sents)):
        x[i] = sent2embedding(sents[i], maxlen_sent, maxlen_word, character_dict)
        y[i] = sent_tags2embedding(tags[i], maxlen_sent, tag_dict, tag_dim)
    return x, y, character_dim, tag_dim


def sent2embedding(sent, maxlen_sent, maxlen_word, character_dict):
    embedding = np.zeros((maxlen_sent, maxlen_word))
    for i, word in enumerate(sent):
        for j, char in enumerate(word):
            embedding[i, j] = character_dict[char]
    return embedding


def sent_tags2embedding(oznake_stavka, maxlen_sent, tag_dict, tag_dim):
    embedding = np.zeros((maxlen_sent, tag_dim))
    for i, tag in enumerate(oznake_stavka):
        embedding[i] = tag_dict[tag]
    return embedding


def build_model(x_sample, maxlen_word, maxlen_sent, character_dim, tag_dim):
    embeddings_size = 60

    embeddings = Sequential()
    embeddings.add(
        Embedding(
            character_dim+1,
            embeddings_size,
            input_length=maxlen_word)
        )
    embeddings.trainable = True

    filter_widths_per_layer = list(range(1, 8))
    num_filters_per_layer = [min(200, 50*w) for w in filter_widths_per_layer]

    inputs = Input(shape=(maxlen_word, embeddings_size))
    L = []
    for n in range(len(num_filters_per_layer)):
        num_filters = num_filters_per_layer[n]
        w = filter_widths_per_layer[n]
        x = Conv1D(num_filters,
                   w,
                   activation='tanh',
                   padding='causal')(inputs)
        L.append(x)

    Convolutional = Model(
        inputs=inputs,
        outputs=L)

    embeddings_output = embeddings.predict(x_sample)
    convolutional_output = Convolutional.predict(embeddings_output)

    inputs = []
    L = []
    for n in range(len(convolutional_output)):
        inputs.append(
            Input(
                shape=(
                    convolutional_output[n].shape[1],
                    convolutional_output[n].shape[2]))
        )
        x = GlobalMaxPool1D()(inputs[n])
        L.append(x)

    outputs = Concatenate()(L)
    MaxPoolingOverTime = Model(
        inputs=inputs,
        outputs=outputs)

    maxpooling_output = MaxPoolingOverTime.predict(convolutional_output)
    inputs = Input(shape=(maxpooling_output.shape[1],))
    transform_gate = Dense(maxpooling_output.shape[1],
                           activation='sigmoid')(inputs)

    carry_gate = Lambda(lambda x: 1-x)(transform_gate)

    z = Add()([
        Multiply()([
            transform_gate,
            Dense(maxpooling_output.shape[1],
                  activation='relu')(inputs)
        ]),
        Multiply()([carry_gate, inputs])
    ])

    Highway = Model(inputs=inputs, outputs=z)
    inputs = embeddings.inputs
    x = embeddings(inputs=inputs)
    x = Convolutional(inputs=x)
    x = MaxPoolingOverTime(inputs=x)
    x = Highway(inputs=x)

    FeatureExtract = Model(inputs=inputs, outputs=x)
    hidden_units = 900

    CharRNN = Sequential()
    CharRNN.add(Masking(mask_value=0., input_shape=(maxlen_sent,
                                                    maxlen_word)))
    CharRNN.add(
        TimeDistributed(
            FeatureExtract,
            input_shape=(maxlen_sent, maxlen_word)
        )
    )

    CharRNN.add(
        Bidirectional(LSTM(
            hidden_units,
            return_sequences=True,
            dropout=0.5,
            recurrent_dropout=0.5
        ))
    )

    CharRNN.add(
        Dropout(0.5)
    )

    CharRNN.add(
        Bidirectional(LSTM(
            hidden_units,
            return_sequences=True
        ))
    )

    CharRNN.add(
        Dropout(0.5)
    )

    CharRNN.add(
        TimeDistributed(
            Dense(tag_dim, activation='sigmoid')
        )
    )

    return CharRNN


def save_model(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def prediction2tags(pred, length, all_tags):
    binary = (pred[:length] > 0.5) * 1
    sent = np.zeros(binary.shape[0])
    for j in range(len(binary)):
        b = binary[j]
        try:
            sent[j] = all_tags.index(embedding2tag(b))
        except ValueError:
            sent[j] = -1
    return sent


def main():
    # Omejitve dolzin besed in povedi
    maxlen_word = 20
    maxlen_sent = 70

    # Podatke preberemo iz datoteke vert
    sents, tags = read_vert_file('./podatki/besede_oznake.tsv')

    # Odstranimo predolge povedi
    sents, tags = filter_sents(sents, tags, maxlen_word, maxlen_sent)

    # Povedi in oznake problikujemo v vektorsko obliko
    x, y, character_dim, tag_dim = load_data(sents, tags, maxlen_word,
                                             maxlen_sent)

    # Zgradimo model
    model = build_model(x[0, :, :], maxlen_word, maxlen_sent, character_dim,
                        tag_dim)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Shranimo konfiguracijo modela
    save_model(model, 'model.json')

    # Naucimo model
    model.fit(x[1:], y[1:], epochs=30)

    # Shranimo utezi modela
    model.save_weights('model.h5')

    # Napovemo oznake testne povedi
    y_predicted = model.predict(x[0:1])

    all_tags = []
    with open('./podatki/oznake_vlozitve_2') as in_file:
        for line in in_file:
            all_tags.append(line.strip().split(', ')[0])
    prediction = prediction2tags(y_predicted[0], len(sents[0]), all_tags)

    print(tags[0])
    print([all_tags[p] if p != -1 else 'Invalid' for p in prediction])


if __name__ == '__main__':
    main()
