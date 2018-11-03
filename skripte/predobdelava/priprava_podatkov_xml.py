import xml.etree.cElementTree as ET
from copy import deepcopy
from random import shuffle
import random

MAXLEN_SENTENCE = 70
MAXLEN_WORD = 20

K_FOLDS = 10
FOLD_WIDTH = 2762
TOTAL_SAMPLES = 27624


DATA_DIR = './podatki/'

random.seed(13)


def normalize(tree, maxlen_sentence, maxlen_word):
    """
    Pretvori dokument v obliki XML-TEI iz kurpusa ssj500k 2.0 s
    slovenskimi oznakami v obliko, zdruzljivo z oznacevalnikom
    Obeliks."""

    # Odstrani div/bibl
    divs = tree.getroot().findall('div')
    for div in divs:
        for bibl in div.findall('bibl'):
            div.remove(bibl)

    # Splosci elemente seg
    for sent in tree.getroot().findall('div/p/s'):
        inserts = []
        for i, token in enumerate(sent):
            if token.tag == 'seg':
                inserts.append((i, token))
        padding = 0
        for ins in inserts[::-1]:
            for j, seg_item in enumerate(ins[1]):
                sent.insert(ins[0] + j + padding, seg_item)
            sent.remove(ins[1])
            padding = len(ins[1]) - 1

    # Preimenuj msd atribute
    for word in tree.getroot().findall('div/p/s/w'):
        msd = word.attrib['ana'].split(':')[1]
        word.attrib.pop('ana')
        word.attrib['msd'] = msd

    # Odstrani linkGrp
    for sent in tree.getroot().findall('div/p/s'):
        for linkGrp in sent.findall('linkGrp'):
            sent.remove(linkGrp)

    # Odstrani presledke, ki niso znotraj povedi.
    for p in tree.getroot().findall('div/p'):
        for space in p.findall('c'):
            p.remove(space)

    # c -> S
    for c in tree.getroot().findall('div/p/s/c'):
        c.text = None
        c.tag = 'S'

    # pc -> c
    for pc in tree.getroot().findall('div/p/s/pc'):
        pc.tag = 'c'
        pc.attrib.pop('ana')

    # Odstrani predolge povedi
    for p in tree.getroot().findall('div/p'):
        for s in p.findall('s'):
            if len(s.findall('w')) + len(s.findall('c')) > maxlen_sentence:
                p.remove(s)

    # Odstrani povedi s predolgimi besedami
    for p in tree.getroot().findall('div/p'):
        for s in p.findall('s'):
            lens = [len(w.text) for w in s.findall('w')]
            if len(lens) != 0 and max(lens) > maxlen_word:
                p.remove(s)

    # Shrani
    return tree


def div_index(div):
    return int(list(div.attrib.values())[0].split('ssj')[1])


def p_index(p):
    return int(list(p.attrib.values())[0].split('.')[1])


def s_index(s):
    return int(list(s.attrib.values())[0].split('.')[2])


def sents2tree(sents):
    body = ET.Element('body')
    div = ET.SubElement(body, 'div')
    p = ET.SubElement(div, 'p')
    for s in sents:
        p.append(s)
    return body


def k_folds_split(in_xml, fold_size, k_folds):

    tree = ET.parse(in_xml)
    tree = normalize(tree, MAXLEN_SENTENCE, MAXLEN_WORD)
    sentences = tree.findall('div/p/s')
    shuffle(sentences)
    folds = []

    for i in range(k_folds-1):
        folds.append(sentences[i * fold_size:(i+1) * fold_size])
    folds.append(sentences[(k_folds-1)*fold_size:])

    for i in range(k_folds):
        body = sents2tree(folds[i])
        ET.ElementTree(body).write(DATA_DIR + 'fold{}_test.xml'.format(i+1),
                                   encoding='UTF-8', xml_declaration=True)
        train = [stavek for j, fold in enumerate(folds)
                 for stavek in fold if j != i]
        body = sents2tree(train)
        ET.ElementTree(body).write(DATA_DIR + 'fold{}_train.xml'.format(i+1),
                                   encoding='UTF-8', xml_declaration=True)


def remove_trailing_elements(tree, last_s_index):
    parent_map = {c: p for p in tree.getroot().iter() for c in p}
    s = tree.getroot().findall('div/p/s')[last_s_index]

    parent_p = parent_map[s]
    parent_div = parent_map[parent_p]

    # Odstrani kasnejše elemente div
    last_div = div_index(parent_div)
    for div in tree.getroot().findall('div'):
        if div_index(div) > last_div:
            parent_map[div].remove(div)

    # Odstrani kasnejše elemente p
    last_p = p_index(parent_p)
    for p in tree.getroot().findall('div/p'):
        if p_index(p) > last_p:
            parent_map[p].remove(p)

    # Odstrani kasnejše elemente s
    last_s = s_index(s)
    for s in tree.getroot().findall('div/p/s'):
        if s_index(s) > last_s:
            parent_map[s].remove(s)

    return tree


def remove_leading_elements(tree, first_s_index):
    parent_map = {c: p for p in tree.getroot().iter() for c in p}

    s = tree.getroot().findall('div/p/s')[first_s_index]
    parent_p = parent_map[s]
    parent_div = parent_map[parent_p]

    # Odstrani prejšnje elemente div
    first_div = div_index(parent_div)
    for div in tree.getroot().findall('div'):
        if div_index(div) < first_div:
            parent_map[div].remove(div)

    # Odstrani kasnejše elemente p
    first_p = p_index(parent_p)
    for p in tree.getroot().findall('div/p'):
        if p_index(p) < first_p:
            parent_map[p].remove(p)

    # Odstrani kasnejše elemente s
    first_s = s_index(s)
    for s in tree.getroot().findall('div/p/s'):
        if s_index(s) < first_s:
            parent_map[s].remove(s)

    return tree


def remove_empty(tree):
    parent_map = {c: p for p in tree.getroot().iter() for c in p}
    for p in tree.getroot().findall('div/p'):
        if len(p.findall('s')) == 0:
            parent_map[p].remove(p)

    for div in tree.getroot().findall('div'):
        if len(div.findall('p')) == 0:
            parent_map[div].remove(div)
    return tree


def remove_between(tree, begin, end):
    parent_map = {c: p for p in tree.getroot().iter() for c in p}

    first_s = tree.getroot().findall('div/p/s')[begin]
    first_p = parent_map[first_s]
    first_div = parent_map[first_p]

    last_s = tree.getroot().findall('div/p/s')[end]
    last_p = parent_map[last_s]
    last_div = parent_map[last_p]

    # Odstrani elemente div
    first_div_i = div_index(first_div)
    last_div_i = div_index(last_div)
    for div in tree.getroot().findall('div'):
        div_i = div_index(div)
        if div_i > first_div_i and div_i < last_div_i:
            parent_map[div].remove(div)

    # Odstrani elemente p
    first_p_i = p_index(first_p)
    last_p_i = p_index(last_p)
    for p in tree.getroot().findall('div/p'):
        p_i = p_index(p)
        if p_i > first_p_i and p_i < last_p_i:
            parent_map[p].remove(p)

    first_s_i = s_index(first_s)
    last_s_i = s_index(last_s)
    for s in tree.getroot().findall('div/p/s'):
        s_i = s_index(s)
        if s_i >= first_s_i and s_i <= last_s_i:
            parent_map[s].remove(s)

    return tree


def split(in_xml, train_xml, test_xml, testfold_begin, testfold_end):
    train_tree = ET.parse(in_xml)
    test_tree = deepcopy(train_tree)
    print('{} {} {}'.format(
        len(test_tree.getroot().findall('div/p/s')),
        testfold_begin,
        testfold_end))

    """ Pripravi testno mnozico """
    test_tree = remove_trailing_elements(test_tree, testfold_end)
    test_tree = remove_leading_elements(test_tree, testfold_begin)
    test_tree = remove_empty(test_tree)
    print(len(test_tree.getroot().findall('div/p/s')))
    test_tree.write(test_xml, encoding='UTF-8', xml_declaration=True)

    """ Pripravi ucno mnozico """
    train_tree = remove_between(train_tree, testfold_begin, testfold_end)
    train_tree = remove_empty(train_tree)
    print(len(train_tree.getroot().findall('div/p/s')))
    train_tree.write(train_xml, encoding='UTF-8', xml_declaration=True)


def main():
    k_folds_split(DATA_DIR+'ssj500k20.xml',
                  FOLD_WIDTH, K_FOLDS)


if __name__ == '__main__':
    main()
