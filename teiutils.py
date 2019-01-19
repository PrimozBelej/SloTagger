import xml.etree.cElementTree as et


def write(path, sentences, tags, lemmas):
    #tree = ET.element(a)
    #for sentence_index, sentence in enumerate(sentences):
    pass


def update_tags(sentence, tags):
    tokens = [child for child in sentence]
    assert len(tokens) == len(tags)
    for token_index, token in enumerate(tokens):
        token.set('ana', 'msd:{}'.format(tags[token_index]))


def add_node(parent, name, text, attributes):
    if parent is not None:
        node = et.SubElement(parent, name)
    else:
        node = et.Element(name)

    if text is not None:
        node.text = text

    if attributes is not None:
        for attribute_name, attribute_value in attributes.items():
            node.set(attribute_name, attribute_value)
    return node


def add_token(sentence_node, token, lemma, tag, xml_id, slo):
    if (slo and tag == 'U') or (not slo and tag == 'Z'):
        token = add_node(
            sentence_node, 'pc', token, {'ana': 'msd:{}'.format(tag)})
    else:
        attributes = {}
        if lemma is not None:
            attributes['lemma'] = lemma
        attributes['ana'] = 'msd:{}'.format(tag)
        if xml_id is not None:
            attributes['xml:id'] = xml_id
        token = add_node(sentence_node, 'w', token, attributes)
    return token


def add_sentence(collection, tokens, lemmas, tags, sentence_index, doc_name, slo):
    assert len(tokens) == len(lemmas)
    assert len(tokens) == len(tags)

    div_id = '{}{}'.format(doc_name, sentence_index+1)
    div = add_node(collection, 'div', None,
                   {'xml:id': div_id})
    paragraph = add_node(div, 'p', None, {'xml:id': div_id+'.1'})
    sentence = add_node(paragraph, 's', None, {'xml:id': div_id+'.1.1'})
    for token_index, token in enumerate(tokens):
        add_token(sentence, token, lemmas[token_index], tags[token_index],
                  div_id + '.1.1.t{}'.format(token_index+1), slo)
    return sentence


def create_document()
    body = add_node(None, 'body', None,
                    {
                        'xmlns': 'http://www.tei-c.org/ns/1.0',
                        'xmlns:xi': 'http://www.w3.org/2001/XInclude',
                        'xml:lang': 'sl'})
    return body


if __name__ == '__main__':
    body = create_document()
    sentence = add_sentence(body, ['"', 'Tistega'], [None, 'tisti'], ['Z', 'Pd-msg'], 0,
                 'test', False)
    print(et.dump(body))

