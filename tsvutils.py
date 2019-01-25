def read(filename, fields=(0, 2)):
    result = [[] for f in fields]
    sentence_values = [[] for f in fields]
    for row_index, row in enumerate(open(filename).readlines()):
        if not row.strip():
            for i, value in enumerate(sentence_values):
                result[i].append(value)
            sentence_values = [[] for f in fields]
            continue

        split_row = row.strip().split('\t')
        if len(split_row) < max(fields)+1:
            raise IndexError("Unexpected number of columns in row {} of file "
                             "{}.".format(row_index, filename))
        for field_index, field in enumerate(fields):
            sentence_values[field_index].append(split_row[field].strip())

    if sentence_values[0]:
        for i, value in enumerate(sentence_values):
            result[i].append(value)

    return result


def write(path, sentences, lemmas, tags):
    if lemmas is not None:
        assert len(sentences) == len(lemmas)
    assert len(sentences) == len(tags)
    with open(path, 'w') as outfile:
        for sentence_index, sentence in enumerate(sentences):
            sentence_tags = tags[sentence_index]
            assert len(sentence_tags) == len(sentence)
            if lemmas is not None:
                sentence_lemmas = lemmas[sentence_index]
                assert len(sentence_lemmas) == len(sentence)
            for word_index, word in enumerate(sentence):
                outfile.write(word)
                outfile.write('\t')
                if lemmas is not None:
                    outfile.write(sentence_lemmas[word_index])
                    outfile.write('\t')
                outfile.write(sentence_tags[word_index])
                outfile.write('\n')
            outfile.write('\n')

