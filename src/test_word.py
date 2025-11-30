from history import State, History
import history
import random
import sys

def S(label, dir = 0, separate = True):
    return State(0, label, dir, separate)

# Dicionário usado para testes:
# https://www.ime.usp.br/~pf/dicios/

LETTERS = {
    'a': S('a'),
    'b': S('b'),
    'c': S('c'),
    'd': S('d'),
    'e': S('e'),
    'f': S('f'),
    'g': S('g'),
    'h': [S('k'), S('h')],
    'i': S('i'),
    'j': [S('i'), S('j')],
    'k': [[S('k', history.DIR_UP)],
          [S('k'), S('k', history.DIR_UP, False)]],
    'l': S('l'),
    'm': S('m'),
    'n': S('n'),
    'o': S('o'),
    'p': S('p'),
    'q': S('q'),
    'r': S('r'),
    's': S('s'),
    't': S('t'),
    'u': S('u'),
    'v': S('v'),
    'w': S('w'),
    'x': [[S('x', history.DIR_LEFT)],
          [S('x'), S('x', history.DIR_LEFT, False)],
          [S('x'), S('x', history.DIR_LEFT, False), S('x', 0, False)]],
    'y': S('y'),
    'z': [S('z'),
          S('z', history.DIR_LEFT, False),
          S('z', history.DIR_DOWN_RIGHT, False),
          S('z', history.DIR_LEFT, False)],
    ' ': S('space'),
}

def word_to_timeline(word: str):
    timeline = []
    for c in word:
        if c in LETTERS:
            state = LETTERS[c]
            if type(state) is State:
                timeline.append(state)
            if type(state) is list:
                if type(state[0]) is list:
                    state = random.choice(state)
                    timeline.extend(state)
                else:
                    timeline.extend(state)
    return timeline

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[ERROR] especifique um arquivo txt com palavras a serem testadas')
        exit(1)

    words = []
    filename = sys.argv[1]
    with open(filename) as file:
        words = [line.rstrip() for line in file]
    print(len(words))

    H = History()
    count = 0
    for word in words:
        word = word.lower()
        print(count, word)
        H.timeline = word_to_timeline(word)
        H.update_word()
        if H.word != word:
            print(f'[ERROR] Assertion error: expected "{word}", got "{H.word}"')
            for s in H.timeline:
                print(s)
            exit(1)
        assert(H.word == word)
        count += 1
    print('ok')
