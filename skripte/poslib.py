import numpy as np


atributi = {
    'besedne_vrste': ['N', 'V', 'A', 'R', 'P', 'M', 'S', 'C', 'Q', 'I', 'Y',
                      'X', 'Z'],
    'dolocnosti': ['y', 'n'],
    'naslonskosti': ['y', 'b'],
    'nikalnosti': ['n', 'y'],
    'oblike': ['n', 'u', 'p', 'r', 'f', 'c', 'm'],
    'osebe': ['1', '2', '3', '-'],
    'skloni': ['n', 'g', 'd', 'a', 'l', 'i', '-'],
    'spoli': ['m', 'f', 'n', '-'],
    'spoli_svojine': ['m', 'f', 'n', '-'],
    'stevila': ['s', 'd', 'p', '-'],
    'stevila_svojine': ['s', 'd', 'p', '-'],
    'stopnje': ['p', 'c', 's'],
    'vidi': ['e', 'p', 'b', '-'],
    'vrste_glagol': ['m', 'a'],
    'vrste_neuvrsceno': ['a', 'e', 'h', 'p', 'w', 't', 'f'],
    'vrste_pridevnik': ['p', 'g', 's'],
    'vrste_prislov': ['r', 'g'],
    'vrste_samostalnik': ['c', 'p'],
    'vrste_stevnik': ['s', 'c', 'o', 'p'],
    'vrste_veznik': ['s', 'c'],
    'vrste_zaimek': ['g', 'd', 'i', 'z', 'p', 'r', 'x', 's', 'q'],
    'zapisi': ['d', 'l', 'r'],
    'zivosti': ['n', 'y']}


vrstni_red = sorted(atributi.keys())
dolzine_vlozitev = dict(zip(vrstni_red, [len(atributi[atr])
                                         if atr == 'besedne_vrste'
                                         else len(atributi[atr]) + 1
                                         for atr in vrstni_red]))
kazalo_vlozitve = dict(zip(vrstni_red,
                           np.cumsum([0] + [dolzine_vlozitev[atr]
                                            for atr in vrstni_red[:-1]])))
kazalo_vlozitve = {atr: (zacetek, zacetek+dolzine_vlozitev[atr])
                   for atr, zacetek in kazalo_vlozitve.items()}
len_vlozitve = sum(dolzine_vlozitev.values())


def funfun(kategorija, vlozitev):
    if vlozitev[-1] == 1:
        return ''
    else:
        return atributi[kategorija][np.argmax(vlozitev)]


def samostalnik_v_oznako(vlozitev):
    oznaka = []

    oznaka += [
        funfun('vrste_samostalnik',
               vlozitev[kazalo_vlozitve['vrste_samostalnik'][0]:
                        kazalo_vlozitve['vrste_samostalnik'][1]])]
    oznaka += [
        funfun('spoli',
               vlozitev[kazalo_vlozitve['spoli'][0]:
                        kazalo_vlozitve['spoli'][1]])]
    oznaka += [
        funfun('stevila',
               vlozitev[kazalo_vlozitve['stevila'][0]:
                        kazalo_vlozitve['stevila'][1]])]
    oznaka += [
        funfun('skloni',
               vlozitev[kazalo_vlozitve['skloni'][0]:
                        kazalo_vlozitve['skloni'][1]])]
    oznaka += [
        funfun('zivosti',
               vlozitev[kazalo_vlozitve['zivosti'][0]:
                        kazalo_vlozitve['zivosti'][1]])]
    return oznaka


def glagol_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_glagol',
               vlozitev[kazalo_vlozitve['vrste_glagol'][0]:
                        kazalo_vlozitve['vrste_glagol'][1]])]
    oznaka += [
        funfun('vidi',
               vlozitev[kazalo_vlozitve['vidi'][0]:
                        kazalo_vlozitve['vidi'][1]])]
    oznaka += [
        funfun('oblike',
               vlozitev[kazalo_vlozitve['oblike'][0]:
                        kazalo_vlozitve['oblike'][1]])]
    oznaka += [
        funfun('osebe',
               vlozitev[kazalo_vlozitve['osebe'][0]:
                        kazalo_vlozitve['osebe'][1]])]
    oznaka += [
        funfun('stevila',
               vlozitev[kazalo_vlozitve['stevila'][0]:
                        kazalo_vlozitve['stevila'][1]])]
    oznaka += [
        funfun('spoli',
               vlozitev[kazalo_vlozitve['spoli'][0]:
                        kazalo_vlozitve['spoli'][1]])]
    oznaka += [
        funfun('nikalnosti',
               vlozitev[kazalo_vlozitve['nikalnosti'][0]:
                        kazalo_vlozitve['nikalnosti'][1]])]
    return oznaka


def pridevnik_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_pridevnik',
               vlozitev[kazalo_vlozitve['vrste_pridevnik'][0]:
                        kazalo_vlozitve['vrste_pridevnik'][1]])]
    oznaka += [
        funfun('stopnje',
               vlozitev[kazalo_vlozitve['stopnje'][0]:
                        kazalo_vlozitve['stopnje'][1]])]
    oznaka += [
        funfun('spoli',
               vlozitev[kazalo_vlozitve['spoli'][0]:
                        kazalo_vlozitve['spoli'][1]])]
    oznaka += [
        funfun('stevila',
               vlozitev[kazalo_vlozitve['stevila'][0]:
                        kazalo_vlozitve['stevila'][1]])]
    oznaka += [
        funfun('skloni',
               vlozitev[kazalo_vlozitve['skloni'][0]:
                        kazalo_vlozitve['skloni'][1]])]
    oznaka += [
        funfun('dolocnosti',
               vlozitev[kazalo_vlozitve['dolocnosti'][0]:
                        kazalo_vlozitve['dolocnosti'][1]])]
    return oznaka


def prislov_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_prislov',
               vlozitev[kazalo_vlozitve['vrste_prislov'][0]:
                        kazalo_vlozitve['vrste_prislov'][1]])]
    oznaka += [
        funfun('stopnje',
               vlozitev[kazalo_vlozitve['stopnje'][0]:
                        kazalo_vlozitve['stopnje'][1]])]
    return oznaka


def zaimek_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_zaimek',
               vlozitev[kazalo_vlozitve['vrste_zaimek'][0]:
                        kazalo_vlozitve['vrste_zaimek'][1]])]
    oznaka += [
        funfun('osebe',
               vlozitev[kazalo_vlozitve['osebe'][0]:
                        kazalo_vlozitve['osebe'][1]])]
    oznaka += [
        funfun('spoli',
               vlozitev[kazalo_vlozitve['spoli'][0]:
                        kazalo_vlozitve['spoli'][1]])]
    oznaka += [
        funfun('stevila',
               vlozitev[kazalo_vlozitve['stevila'][0]:
                        kazalo_vlozitve['stevila'][1]])]
    oznaka += [
        funfun('skloni',
               vlozitev[kazalo_vlozitve['skloni'][0]:
                        kazalo_vlozitve['skloni'][1]])]
    oznaka += [
        funfun('stevila_svojine',
               vlozitev[kazalo_vlozitve['stevila_svojine'][0]:
                        kazalo_vlozitve['stevila_svojine'][1]])]
    oznaka += [
        funfun('spoli_svojine',
               vlozitev[kazalo_vlozitve['spoli_svojine'][0]:
                        kazalo_vlozitve['spoli_svojine'][1]])]
    oznaka += [
        funfun('naslonskosti',
               vlozitev[kazalo_vlozitve['naslonskosti'][0]:
                        kazalo_vlozitve['naslonskosti'][1]])]
    return oznaka


def stevnik_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('zapisi',
               vlozitev[kazalo_vlozitve['zapisi'][0]:
                        kazalo_vlozitve['zapisi'][1]])]
    oznaka += [
        funfun('vrste_stevnik',
               vlozitev[kazalo_vlozitve['vrste_stevnik'][0]:
                        kazalo_vlozitve['vrste_stevnik'][1]])]
    oznaka += [
        funfun('spoli',
               vlozitev[kazalo_vlozitve['spoli'][0]:
                        kazalo_vlozitve['spoli'][1]])]
    oznaka += [
        funfun('stevila',
               vlozitev[kazalo_vlozitve['stevila'][0]:
                        kazalo_vlozitve['stevila'][1]])]
    oznaka += [
        funfun('skloni',
               vlozitev[kazalo_vlozitve['skloni'][0]:
                        kazalo_vlozitve['skloni'][1]])]
    oznaka += [
        funfun('dolocnosti',
               vlozitev[kazalo_vlozitve['dolocnosti'][0]:
                        kazalo_vlozitve['dolocnosti'][1]])]
    return oznaka


def predlog_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('skloni',
               vlozitev[kazalo_vlozitve['skloni'][0]:
                        kazalo_vlozitve['skloni'][1]])]
    return oznaka


def veznik_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_veznik',
               vlozitev[kazalo_vlozitve['vrste_veznik'][0]:
                        kazalo_vlozitve['vrste_veznik'][1]])]
    return oznaka


def neuvrsceno_v_oznako(vlozitev):
    oznaka = []
    oznaka += [
        funfun('vrste_neuvrsceno',
               vlozitev[kazalo_vlozitve['vrste_neuvrsceno'][0]:
                        kazalo_vlozitve['vrste_neuvrsceno'][1]])]
    return oznaka


preslikave_emb2pos = {'N': samostalnik_v_oznako,
                      'V': glagol_v_oznako,
                      'A': pridevnik_v_oznako,
                      'R': prislov_v_oznako,
                      'P': zaimek_v_oznako,
                      'M': stevnik_v_oznako,
                      'S': predlog_v_oznako,
                      'C': veznik_v_oznako,
                      'X': neuvrsceno_v_oznako}


def vlozitev_v_oznako(vlozitev):
    oznaka = []
    i = np.argmax(vlozitev[kazalo_vlozitve['besedne_vrste'][0]:
                           kazalo_vlozitve['besedne_vrste'][1]])
    oznaka += [atributi['besedne_vrste'][i]]
    vrsta = oznaka[0]
    preslikava = preslikave_emb2pos.get(vrsta)
    if preslikava is None:
        pass
    else:
        oznaka += preslikava(vlozitev)

    return ''.join(oznaka)


def samostalnik_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1
    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_samostalnik'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    atr = 'spoli'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
    neg -= {atr}

    atr = 'stevila'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[3])] = 1
    neg -= {atr}

    atr = 'skloni'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[4])] = 1
    neg -= {atr}

    if len(oznaka) > 5:
        atr = 'zivosti'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[5])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def glagol_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1
    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_glagol'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    atr = 'vidi'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
    neg -= {atr}

    atr = 'oblike'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[3])] = 1
    neg -= {atr}

    if len(oznaka) > 4:
        atr = 'osebe'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[4])] = 1
        neg -= {atr}
    if len(oznaka) > 5:
        atr = 'stevila'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[5])] = 1
        neg -= {atr}
    if len(oznaka) > 6:
        atr = 'spoli'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[6])] = 1
        neg -= {atr}
    if len(oznaka) > 7:
        atr = 'nikalnosti'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[7])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def pridevnik_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_pridevnik'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    atr = 'stopnje'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
    neg -= {atr}

    atr = 'spoli'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[3])] = 1
    neg -= {atr}

    atr = 'stevila'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[4])] = 1
    neg -= {atr}

    atr = 'skloni'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[5])] = 1
    neg -= {atr}

    if len(oznaka) > 6:
        atr = 'dolocnosti'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[6])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def prislov_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_prislov'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    if len(oznaka) > 2:
        atr = 'stopnje'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def zaimek_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_zaimek'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    if len(oznaka) > 2:
        atr = 'osebe'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
        neg -= {atr}

    if len(oznaka) > 3:
        atr = 'spoli'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[3])] = 1
        neg -= {atr}

    if len(oznaka) > 4:
        atr = 'stevila'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[4])] = 1
        neg -= {atr}

    if len(oznaka) > 5:
        atr = 'skloni'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[5])] = 1
        neg -= {atr}

    if len(oznaka) > 6:
        atr = 'stevila_svojine'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[6])] = 1
        neg -= {atr}

    if len(oznaka) > 7:
        atr = 'spoli_svojine'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[7])] = 1
        neg -= {atr}

    if len(oznaka) > 8:
        atr = 'naslonskosti'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[8])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def stevnik_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1
    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'zapisi'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    atr = 'vrste_stevnik'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[2])] = 1
    neg -= {atr}

    if len(oznaka) > 3:
        atr = 'spoli'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[3])] = 1
        neg -= {atr}

    if len(oznaka) > 4:
        atr = 'stevila'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[4])] = 1
        neg -= {atr}

    if len(oznaka) > 5:
        atr = 'skloni'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[5])] = 1
        neg -= {atr}

    if len(oznaka) > 6:
        atr = 'dolocnosti'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[6])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def predlog_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'skloni'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def veznik_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    atr = 'vrste_veznik'
    vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
    neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def neuvrsceno_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    if len(oznaka) > 1:
        atr = 'vrste_neuvrsceno'
        vlozitev[kazalo_vlozitve[atr][0] + atributi[atr].index(oznaka[1])] = 1
        neg -= {atr}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


def ostalo_v_vlozitev(oznaka):
    vlozitev = np.zeros(len_vlozitve, dtype=int)
    vrsta = oznaka[0]
    vlozitev[atributi['besedne_vrste'].index(vrsta)] = 1

    neg = set(atributi.keys()) - {'besedne_vrste'}

    # Nastavi NA
    for n in neg:
        vlozitev[kazalo_vlozitve[n][1] - 1] = 1

    return vlozitev


preslikave_pos2emb = {'N': samostalnik_v_vlozitev,
                      'V': glagol_v_vlozitev,
                      'A': pridevnik_v_vlozitev,
                      'R': prislov_v_vlozitev,
                      'P': zaimek_v_vlozitev,
                      'M': stevnik_v_vlozitev,
                      'S': predlog_v_vlozitev,
                      'C': veznik_v_vlozitev,
                      'Q': ostalo_v_vlozitev,
                      'I': ostalo_v_vlozitev,
                      'Y': ostalo_v_vlozitev,
                      'X': neuvrsceno_v_vlozitev,
                      'Z': ostalo_v_vlozitev
                      }


def oznaka_v_vlozitev(oznaka):

    vrsta = oznaka[0]
    preslikava = preslikave_pos2emb.get(vrsta)
    if preslikava is None:
        return None
    else:
        return preslikava(oznaka)
