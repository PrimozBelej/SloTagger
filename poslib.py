import numpy as np


PROPERTIES = {
    'category': ['N', 'V', 'A', 'R', 'P', 'M', 'S', 'C', 'Q', 'I', 'Y',
                 'X', 'Z'],
    'definiteness': ['y', 'n'],
    'clitic': ['y', 'b'],
    'negative': ['n', 'y'],
    'vform': ['n', 'u', 'p', 'r', 'f', 'c', 'm'],
    'person': ['1', '2', '3', '-'],
    'case': ['n', 'g', 'd', 'a', 'l', 'i', '-'],
    'gender': ['m', 'f', 'n', '-'],
    'owner_gender': ['m', 'f', 'n', '-'],
    'number': ['s', 'd', 'p', '-'],
    'owner_number': ['s', 'd', 'p', '-'],
    'degree': ['p', 'c', 's'],
    'aspect': ['e', 'p', 'b', '-'],
    'type_verb': ['m', 'a'],
    'type_residual': ['a', 'e', 'h', 'p', 'w', 't', 'f'],
    'type_adjective': ['p', 'g', 's'],
    'type_adverb': ['r', 'g'],
    'type_noun': ['c', 'p'],
    'type_numeral': ['s', 'c', 'o', 'p'],
    'type_conjunction': ['s', 'c'],
    'type_pronoun': ['g', 'd', 'i', 'z', 'p', 'r', 'x', 's', 'q'],
    'form': ['d', 'l', 'r'],
    'animate': ['n', 'y']}

CATEGORY_PROPERTIES = {
    'N': ('type_noun', 'gender', 'number', 'case', 'animate'),
    'V': ('type_verb', 'aspect', 'vform', 'person', 'number', 'gender',
          'negative'),
    'A': ('type_adjective', 'degree', 'gender', 'number', 'case',
          'definiteness'),
    'R': ('type_adverb', 'degree'),
    'P': ('type_pronoun', 'person', 'gender', 'number', 'case', 'owner_number',
          'owner_gender', 'clitic'),
    'M': ('form', 'type_numeral', 'gender', 'number', 'case', 'definiteness'),
    'S': ('case'),
    'C': ('type_conjunction'),
    'Q': (),
    'I': (),
    'Y': (),
    'X': ('type_residual'),
    'Z': ()
}

PROPERTY_ORDER = sorted(PROPERTIES.keys())

PROPERTY_LENGTHS = dict(zip(
    PROPERTY_ORDER,
    [len(PROPERTIES[atr]) if atr == 'category' else len(PROPERTIES[atr]) + 1
     for atr in PROPERTY_ORDER]))

PROPERTY_INDEX = dict(zip(
    PROPERTY_ORDER,
    np.cumsum([0] + [PROPERTY_LENGTHS[atr] for atr in PROPERTY_ORDER[:-1]])))

PROPERTY_INDEX = {
    atr: (zacetek, zacetek+PROPERTY_LENGTHS[atr])
    for atr, zacetek in PROPERTY_INDEX.items()}

EMBEDDING_LENGTH = sum(PROPERTY_LENGTHS.values())


def property2embedding(property_, embedding):
    index = PROPERTY_INDEX[property_]
    property_embedding = embedding[index[0]:index[1]]
    if property_embedding[-1] == 1:
        return ''
    else:
        return PROPERTIES[property_][np.argmax(property_embedding)]


def embedding2tag(embedding):
    tag = []
    i = np.argmax(embedding[PROPERTY_INDEX['category'][0]:
                            PROPERTY_INDEX['category'][1]])
    tag += [PROPERTIES['category'][i]]
    category = tag[0]
    for property_ in CATEGORY_PROPERTIES[category]:
        tag.append(property2embedding(property_, embedding))
    return ''.join(tag)


def noun2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1
    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_noun'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    atr = 'gender'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
    neg -= {atr}

    atr = 'number'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[3])] = 1
    neg -= {atr}

    atr = 'case'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[4])] = 1
    neg -= {atr}

    if len(tag) > 5:
        atr = 'animate'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[5])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def verb2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1
    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_verb'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    atr = 'aspect'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
    neg -= {atr}

    atr = 'vform'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[3])] = 1
    neg -= {atr}

    if len(tag) > 4:
        atr = 'person'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[4])] = 1
        neg -= {atr}
        if len(tag) > 5:
            atr = 'number'
            embedding[PROPERTY_INDEX[atr][0] +
                      PROPERTIES[atr].index(tag[5])] = 1
            neg -= {atr}
            if len(tag) > 6:
                atr = 'gender'
                embedding[PROPERTY_INDEX[atr][0] +
                          PROPERTIES[atr].index(tag[6])] = 1
                neg -= {atr}
                if len(tag) > 7:
                    atr = 'negative'
                    embedding[PROPERTY_INDEX[atr][0] +
                              PROPERTIES[atr].index(tag[7])] = 1
                    neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def adjective2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_adjective'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    atr = 'degree'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
    neg -= {atr}

    atr = 'gender'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[3])] = 1
    neg -= {atr}

    atr = 'number'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[4])] = 1
    neg -= {atr}

    atr = 'case'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[5])] = 1
    neg -= {atr}

    if len(tag) > 6:
        atr = 'definiteness'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[6])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def adverb2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_adverb'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    if len(tag) > 2:
        atr = 'degree'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def pronoun2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_pronoun'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    if len(tag) > 2:
        atr = 'person'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
        neg -= {atr}

    if len(tag) > 3:
        atr = 'gender'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[3])] = 1
        neg -= {atr}

    if len(tag) > 4:
        atr = 'number'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[4])] = 1
        neg -= {atr}

    if len(tag) > 5:
        atr = 'case'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[5])] = 1
        neg -= {atr}

    if len(tag) > 6:
        atr = 'owner_number'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[6])] = 1
        neg -= {atr}

    if len(tag) > 7:
        atr = 'owner_gender'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[7])] = 1
        neg -= {atr}

    if len(tag) > 8:
        atr = 'clitic'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[8])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def numeral2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1
    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'form'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    atr = 'type_numeral'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[2])] = 1
    neg -= {atr}

    if len(tag) > 3:
        atr = 'gender'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[3])] = 1
        neg -= {atr}

    if len(tag) > 4:
        atr = 'number'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[4])] = 1
        neg -= {atr}

    if len(tag) > 5:
        atr = 'case'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[5])] = 1
        neg -= {atr}

    if len(tag) > 6:
        atr = 'definiteness'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[6])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def adposition2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'case'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def conjunction2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    atr = 'type_conjunction'
    embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
    neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def residual2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    if len(tag) > 1:
        atr = 'type_residual'
        embedding[PROPERTY_INDEX[atr][0] + PROPERTIES[atr].index(tag[1])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


def other2embedding(tag):
    embedding = np.zeros(EMBEDDING_LENGTH, dtype=int)
    category = tag[0]
    embedding[PROPERTIES['category'].index(category)] = 1

    neg = set(PROPERTIES.keys()) - {'category'}

    # Nastavi NA
    for n in neg:
        embedding[PROPERTY_INDEX[n][1] - 1] = 1

    return embedding


TAG2EMBEDDING_TRANSFORMATIONS = {
    'N': noun2embedding,
    'V': verb2embedding,
    'A': adjective2embedding,
    'R': adverb2embedding,
    'P': pronoun2embedding,
    'M': numeral2embedding,
    'S': adposition2embedding,
    'C': conjunction2embedding,
    'Q': other2embedding,
    'I': other2embedding,
    'Y': other2embedding,
    'X': residual2embedding,
    'Z': other2embedding}


def tag2embedding(tag):
    category = tag[0]
    preslikava = TAG2EMBEDDING_TRANSFORMATIONS.get(category)
    if preslikava is None:
        return None
    return preslikava(tag)
