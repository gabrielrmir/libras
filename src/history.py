import enum
import options
import math

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

class TokenType(enum.Enum):
    LABEL = 0
    DIRECTION = 1

class Token():
    def __init__(self, value, timestamp_sec: float, type: TokenType):
        self.type = type
        self.value = value
        self.time_start = timestamp_sec
        self.time_end = timestamp_sec

    def extend_time(self, timestamp_sec: float):
        if timestamp_sec < self.time_start:
            return
        self.time_end = timestamp_sec

    def get_duration(self):
        return self.time_end-self.time_start

    def is_long_enough(self):
        return self.get_duration() >= options.minimum_duration

class History():
    def __init__(self):
        self.last_label = None
        self.last_direction = None
        self.timeline: list[Token]
        self.timeline = []

    def push_label(self, label: str, timestamp_sec: float):
        if self.last_label and self.last_label.value == label:
            self.last_label.extend_time(timestamp_sec)
            return

        if self.last_label and not self.last_label.is_long_enough():
            self.last_label = None
            for i in range(len(self.timeline)-1, -1, -1):
                if self.timeline[i].type == TokenType.LABEL:
                    self.timeline.pop(i)
                    break

        new_token = Token(label, timestamp_sec, TokenType.LABEL)
        self.timeline.append(new_token)
        self.last_label = new_token

    def push_motion(self, motion: tuple[float,float], timestamp_sec: float):
        x,y = motion
        angle = math.atan2(y,x)-math.pi/8
        octant = int(round(8*angle/(2*math.pi)+8)%8)
        direction = HALF_WIND_MAP[octant]

        if self.last_direction and self.last_direction.value == direction:
            self.last_direction.extend_time(timestamp_sec)
            return

        new_token = Token(direction, timestamp_sec, TokenType.DIRECTION)
        self.timeline.append(new_token)
        self.last_direction = new_token

    def clear(self):
        if len(self.timeline) == 0:
            return
        self.last_label = None
        self.last_direction = None
        self.timeline = []

    def __str__(self):
        s = ''
        for i in self.timeline:
            if i.type == TokenType.LABEL:
                s += f'{i.value}'
        return s
