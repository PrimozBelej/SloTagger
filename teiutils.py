#import xml.etree.cElementTree as et
from lxml import etree as et
from config import MAXLEN_SENTENCE, MAXLEN_WORD


NSMAP = {
    None: 'http://www.tei-c.org/ns/1.0',
    'xi': 'http://www.w3.org/2001/XInclude',
    'lang': 'sl'}


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


def read(path, tags, beginning=0, end=-1):
    tree = et.parse(path)
    root = tree.getroot()
    namespace = root.tag[:root.tag.find('}')+1]
    sentences = tree.getroot().findall('*/'+namespace+'p/'+namespace+'s')
    if end < 0:
        end = len(sentences)
    for sentence_index, sentence in enumerate(sentences):
        if beginning > sentence_index:
            continue
        if end <= sentence_index:
            break
        yield parse_sentence_node(sentence, namespace, tags)


def update_sentence_tags(sentence, tags, namespace):
    tokens = [child for child in sentence
              if child.tag in {namespace+'pc', namespace+'w'}]
    assert len(tokens) == len(tags)
    for token_index, token in enumerate(tokens):
        token.set('ana', 'msd:{}'.format(tags[token_index]))


def update_tags(in_path, out_path, tags, beginning=0):
    tree = et.parse(in_path)
    root = tree.getroot()
    namespace = root.tag[:root.tag.find('}')+1]
    for sentence_index, sentence in enumerate(
        tree.getroot().findall(
            '*/'+namespace+'p/'+namespace+'s')):
        if sentence_index < beginning:
            continue
        if sentence_index >= beginning+len(tags):
            break
        if not tags[sentence_index-beginning]:
            continue
        update_sentence_tags(sentence, tags[sentence_index-beginning], namespace)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

