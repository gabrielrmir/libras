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
        if utils.vec_len(motion) > .1:
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

    def consume(self, i: int, label: str) -> int:
        size = len(self.timeline)
        while i < size and self.timeline[i].label == label:
            i += 1
        return i

    def at(self, i: int) -> State | None:
        if i < len(self.timeline):
            return self.timeline[i]
        return None

    def update_word(self):
        # print(self)
        i = 0
        size = len(self.timeline)
        word = ''
        while i < size:
            s = self.timeline[i]
            if s.label == 'i':
                i = self.consume(i, 'i')
                s = self.at(i)
                if s and s.label == 'j':
                    self.consume(i, 'j')
                    word += 'j'
                else:
                    word += 'i'
                continue
            elif s.label in SIMPLE_LETTERS and s.direction == 0:
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
