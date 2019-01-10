from keras import Sequential, Model
from keras.models import model_from_json
from keras.metrics import categorical_accuracy
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input, Conv1D, Concatenate,\
        GlobalMaxPool1D, Dense, Lambda, Add, Multiply, Masking, Bidirectional,\
        Dropout, TimeDistributed
import keras


MAXLEN_WORD = 70
MAXLEN_SENTENCE = 20
CHAR_EMBEDDING_SIZE = 60


def build_input_network(character_dim):
    model = Sequential()
    model.add(
        Embedding(
            character_dim+1,
            CHAR_EMBEDDING_SIZE,
            input_length=MAXLEN_WORD)
    )
    model.trainable = True
    return model


def build_convolutional_network():
    filter_widths_per_layer = list(range(1, 8))
    num_filters_per_layer = [
        min(200, 50*width) for width in filter_widths_per_layer]
    inputs = Input(shape=(MAXLEN_WORD, CHAR_EMBEDDING_SIZE))
    layers = []
    for index, num_filters in enumerate(num_filters_per_layer):
        width = filter_widths_per_layer[index]
        layer = Conv1D(num_filters,
                       width,
                       activation='tanh',
                       padding='causal')(inputs)
        layers.append(layer)

    convolutional_network = Model(
        inputs=inputs,
        outputs=layers)
    return convolutional_network


def build_pooling_network(convolutional_output):
    inputs = []
    layers = []
    for output_index, output in enumerate(convolutional_output):
        inputs.append(
            Input(
                shape=(
                    output.shape[1],
                    output.shape[2]))
        )
        layer = GlobalMaxPool1D()(inputs[output_index])
        layers.append(layer)

    outputs = Concatenate()(layers)
    pooling_network = Model(
        inputs=inputs,
        outputs=outputs)

    return pooling_network


def build_highway_network(maxpooling_output):
    inputs = Input(shape=(maxpooling_output.shape[1],))
    transform_gate = Dense(maxpooling_output.shape[1],
                           activation='sigmoid')(inputs)
    carry_gate = Lambda(lambda network_input: 1-network_input)(transform_gate)
    network = Add()([
        Multiply()([
            transform_gate,
            Dense(maxpooling_output.shape[1],
                  activation='relu')(inputs)
        ]),
        Multiply()([carry_gate, inputs])
    ])
    return Model(inputs=inputs, outputs=network)


def build_recurrent_network(feature_extract, tag_dim):
    hidden_units = 900
    rnn = Sequential()
    rnn.add(Masking(mask_value=0., input_shape=(MAXLEN_SENTENCE,
                                                MAXLEN_WORD)))
    rnn.add(
        TimeDistributed(
            feature_extract,
            input_shape=(MAXLEN_SENTENCE, MAXLEN_WORD)
        )
    )

    rnn.add(
        Bidirectional(LSTM(
            hidden_units,
            return_sequences=True,
            dropout=0.5,
            recurrent_dropout=0.5
        ))
    )

    rnn.add(
        Dropout(0.5)
    )

    rnn.add(
        Bidirectional(LSTM(
            hidden_units,
            return_sequences=True
        ))
    )

    rnn.add(
        Dropout(0.5)
    )

    rnn.add(
        TimeDistributed(
            Dense(tag_dim, activation='sigmoid')
        )
    )

    return rnn


def build_model(x_sample, character_dim, tag_dim):
    input_network = build_input_network(character_dim)
    embeddings_output = input_network.predict(x_sample)

    convolutional_network = build_convolutional_network()
    convolutional_output = convolutional_network.predict(embeddings_output)

    pooling_network = build_pooling_network(convolutional_output)
    maxpooling_output = pooling_network.predict(convolutional_output)

    highway_network = build_highway_network(maxpooling_output)
    inputs = input_network.inputs
    model = input_network(inputs=inputs)
    model = convolutional_network(inputs=model)
    model = pooling_network(inputs=model)
    model = highway_network(inputs=model)

    feature_extract = Model(inputs=inputs, outputs=model)

    return build_recurrent_network(feature_extract, tag_dim)


def bv_acc(y_true, y_pred):
    return categorical_accuracy(y_true[:, :, :13], y_pred[:, :, :13])


def load_model(configuration_path, weights_path):
    keras.metrics.bv_acc = bv_acc
    json_file = open(configuration_path, 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    model.compile(loss='binary_crossentropy', metrics=[bv_acc],
                  optimizer='adam')
    model.load_weights(weights_path)
    return model


def save_model(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
