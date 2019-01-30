#import xml.etree.cElementTree as et
from lxml import etree as et
from config import MAXLEN_SENTENCE, MAXLEN_WORD


NSMAP = {
    None: 'http://www.tei-c.org/ns/1.0',
    'xi': 'http://www.w3.org/2001/XInclude',
    'lang': 'sl'}

def write(path, sentences, tags):
    body = create_document()
    for sentence_index, sentence in enumerate(sentences):
        add_sentence(body, sentence, [None]*len(sentence),
                     tags[sentence_index], sentence_index,
                     'test', False)
    et.ElementTree(body).write(path, encoding="utf-8", xml_declaration=True,
                               pretty_print=True)


def parse_sentence_node(sentence_node, namespace, return_tags):
    sentence = []
    for child in sentence_node:
        if child.tag not in {namespace+'pc', namespace+'w'}:
            continue
        if len(child.text) > MAXLEN_WORD:
            return []
        if return_tags:
            tag = child.get('ana')
            sentence.append(tag[4:] if tag is not None and len(tag) > 4 else
                            None)
        else:
            sentence.append(child.text)
    if len(sentence) > MAXLEN_SENTENCE:
        return []
    return sentence


def read(path, tags):
    tree = et.parse(path)
    root = tree.getroot()
    namespace = root.tag[:root.tag.find('}')+1]
    for sentence in tree.getroot().findall(
        '*/'+namespace+'p/'+namespace+'s'):
        yield parse_sentence_node(sentence, namespace, tags)


def update_sentence_tags(sentence, tags, namespace):
    tokens = [child for child in sentence
              if child.tag in {namespace+'pc', namespace+'w'}]
    assert len(tokens) == len(tags)
    for token_index, token in enumerate(tokens):
        token.set('ana', 'msd:{}'.format(tags[token_index]))


def update_tags(in_path, out_path, tags):
    tree = et.parse(in_path)
    root = tree.getroot()
    namespace = root.tag[:root.tag.find('}')+1]
    for sentence_index, sentence in enumerate(
        tree.getroot().findall(
            '*/'+namespace+'p/'+namespace+'s')):
        if sentence_index >= len(tags):
            break
        if not tags[sentence_index]:
            continue
        update_sentence_tags(sentence, tags[sentence_index], namespace)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def add_node(parent, name, text, attributes):
    if parent is not None:
        node = et.SubElement(parent, name)
    else:
        node = et.Element(name, nsmap = NSMAP)

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
            attributes['{http://www.w3.org/XML/1998/namespace}id'] = xml_id
        token = add_node(sentence_node, 'w', token, attributes)
    return token


def add_sentence(collection, tokens, lemmas, tags, sentence_index, doc_name, slo):
    assert len(tokens) == len(lemmas)
    assert len(tokens) == len(tags)

    div_id = '{}{}'.format(doc_name, sentence_index+1)
    div = add_node(collection, 'div', None,
                   {'{http://www.w3.org/XML/1998/namespace}id': div_id})
    paragraph = add_node(div, 'p', None,
                         {'{http://www.w3.org/XML/1998/namespace}id':
                          div_id+'.1'})
    sentence = add_node(
        paragraph, 's', None,
        {'{http://www.w3.org/XML/1998/namespace}id': div_id+'.1.1'})
    for token_index, token in enumerate(tokens):
        add_token(sentence, token, lemmas[token_index], tags[token_index],
                  div_id + '.1.1.t{}'.format(token_index+1), slo)
    return sentence


def create_document():
    body = add_node(None, 'body', None,{})
    """
    {
    'xmlns': 'http://www.tei-c.org/ns/1.0',
    'xmlns:xi': 'http://www.w3.org/2001/XInclude',
    'xml:lang': 'sl'
    })
    """
    return body

