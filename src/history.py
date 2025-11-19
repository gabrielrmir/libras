import enum
import options
import math
import utils

# Bitflag com 8 direções cardeais e colaterais. Usado para representar a
# direção dos movimentos. Um movimento pode ser classificado como mais de uma
# possível direção.
# Ex: Movimento para direita levemente inclinado para cima = RIGHT | UP_RIGHT
DIR_RIGHT      = 1
DIR_DOWN_RIGHT = 2
DIR_DOWN       = 2 << 1
DIR_DOWN_LEFT  = 2 << 2
DIR_LEFT       = 2 << 3
DIR_UP_LEFT    = 2 << 4
DIR_UP         = 2 << 5
DIR_UP_RIGHT   = 2 << 6

HALF_WIND_MAP = {
    0: DIR_RIGHT | DIR_DOWN_RIGHT,
    1: DIR_DOWN_RIGHT | DIR_DOWN,
    2: DIR_DOWN | DIR_DOWN_LEFT,
    3: DIR_DOWN_LEFT | DIR_LEFT ,
    4: DIR_LEFT | DIR_UP_LEFT,
    5: DIR_UP_LEFT | DIR_UP,
    6: DIR_UP | DIR_UP_RIGHT,
    7: DIR_UP_RIGHT | DIR_RIGHT,
}

DIR_DISPLAY_NAME = {
    DIR_RIGHT: 'r',
    DIR_DOWN_RIGHT: 'dr',
    DIR_DOWN: 'd',
    DIR_DOWN_LEFT: 'dl',
    DIR_LEFT: 'l',
    DIR_UP_LEFT: 'ul',
    DIR_UP: 'u',
    DIR_UP_RIGHT: 'ur'
}

SIMPLE_LETTERS = 'abcdefglmnopqrstuvwy'

def dirs_to_str(dirs: int):
    s = ''
    i = 1
    while i <= dirs:
        if i & dirs:
            s += f'{DIR_DISPLAY_NAME[i]},'
        i <<= 1
    return s

class State():
    def __init__(self, timestamp_sec: float, label: str, direction: int = 0):
        self.label = label
        self.direction = direction
        self.time_start = timestamp_sec
        self.time_end = timestamp_sec

    def extend_time(self, timestamp_sec: float):
        if timestamp_sec > self.time_start:
            self.time_end = timestamp_sec

    def __str__(self):
        return f'<{self.label}:{dirs_to_str(self.direction)}>'

    def __eq__(self, state):
        return self.label == state.label and \
            self.direction == state.direction

class History():
    def __init__(self):
        self.last_label: str
        self.last_label = ''
        self.last_label_start = 0.0
        self.last_label_end = 0.0

        self.last_direction: int
        self.last_direction = 0

        self.timeline: list[State]
        self.timeline = []

        self.word = ''

    def push_motion(self, timestamp_sec: float, motion: tuple[float, float]):
        direction = 0
        if utils.vec_len(motion) > .2:
            angle = math.atan2(motion[1], motion[0])-math.pi/8
            octant = int(round(8*angle/(2*math.pi)+8)%8)
            direction = HALF_WIND_MAP[octant]
        if direction != self.last_direction:
            self.last_direction = direction
            self.push_state(timestamp_sec)

    def push_label(self, timestamp_sec: float, label: str):
        if self.last_label != label:
            self.last_label = label
            self.last_label_start = timestamp_sec
            self.last_label_end = timestamp_sec
            return
        if timestamp_sec > self.last_label_end:
            self.last_label_end = timestamp_sec
        self.push_state(timestamp_sec)

    def push_state(self, timestamp_sec: float):
        if self.last_label_end - self.last_label_start < options.minimum_duration:
            return

        if self.timeline and self.timeline[-1].label == self.last_label and \
            self.timeline[-1].direction == self.last_direction:
            self.timeline[-1].extend_time(timestamp_sec)
        else:
            state = State(timestamp_sec, self.last_label, self.last_direction)
            self.timeline.append(state)
            self.update_word()

    def consume(self, i: int, label: str, direction: int = -1, optional: bool = False) -> int:
        size = len(self.timeline)
        while i < size and self.timeline[i].label == label and \
            (direction == -1 or self.timeline[i].direction & direction or \
             self.timeline[i].direction == direction or (self.timeline[i].direction == 0 and optional)):
            i += 1
        return i

    def at(self, i: int) -> State | None:
        if i < len(self.timeline):
            return self.timeline[i]
        return None

    def sequence(self, i: int, l1: str, l2: str, push_l1: bool = True):
        s = self.at(i)
        if not s:
            return i, ''

        if not s.label == l1 or s.direction != 0:
            return i, ''

        i = self.consume(i, l1, 0)
        s = self.at(i)
        if s and s.label == l2:
            i = self.consume(i, l2)
            return i, l2

        if push_l1:
            return i, l1

        return i, ''

    def directions(self, i: int, label: str, directions: list[int]):
        i = self.skip(i, label)
        s = self.at(i)
        for dir in directions:
            i = self.skip(i, label)
            s = self.at(i)
            if not s or s.label != label or not s.direction & dir:
                return i, ''
            i = self.consume(i, label, dir)
        i = self.skip(i, label)
        return i, label

    def skip(self, i: int, label: str):
        s = self.at(i)
        while s and s.label == label and s.direction == 0:
            i += 1
            s = self.at(i)
        return i

    def update_word(self):
        print(chr(27) + "[2J")
        for s in self.timeline:
            print(s, end=' ')
        print()

        i = 0
        size = len(self.timeline)
        word = ''
        while i < size:
            s = self.timeline[i]

            i, l = self.sequence(i, 'i', 'j')
            if l:
                word += l
                continue

            i, l = self.sequence(i, 'p', 'k')
            if l:
                word += l
                continue

            i, l = self.sequence(i, 'k', 'h', False)
            if l:
                word += l
                continue

            i, l = self.directions(i, 'k', [DIR_UP])
            if l:
                word += l
                continue

            i, l = self.directions(i, 'x', [DIR_LEFT])
            if l:
                word += l
                continue

            i, l = self.directions(i, 'z', [DIR_LEFT, DIR_DOWN_RIGHT, DIR_LEFT])
            if l:
                print(self.at(i))
                word += l
                continue

            if s.label in SIMPLE_LETTERS and s.direction == 0:
                word += s.label
            i += 1

        self.word = word

    def clear(self):
        self.timeline = []
        self.last_direction = 0
        self.last_label = ''
        self.last_label_start = 0.0
        self.last_label_end = 0.0
        self.word = ''

    def __str__(self):
        states = []
        for state in self.timeline:
            states.append(str(state))
        return ','.join(states)
