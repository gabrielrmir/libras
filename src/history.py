import enum

# Bitflag com 8 direções cardeais e colaterais. Usado para representar a
# direção dos movimentos. Um movimento pode ser classificado como mais de uma
# possível direção.
# Ex: Movimento para direita levemente inclinado para cima = RIGHT | UP_RIGHT
class Dir8(enum.Enum):
    NONE       = 0
    UP         = 1
    RIGHT      = 2
    DOWN       = 2 << 1
    LEFT       = 2 << 2
    UP_RIGHT   = 2 << 3
    DOWN_RIGHT = 2 << 4
    DOWN_LEFT  = 2 << 5
    UP_LEFT    = 2 << 6

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

class History():
    def __init__(self):
        self.last_label = None
        self.last_direction = None
        self.timeline = []

    def push_label(self, label: str, timestamp_sec: float):
        if self.last_label and self.last_label.value == label:
            self.last_label.extend_time(timestamp_sec)
            return
        new_token = Token(label, timestamp_sec, TokenType.LABEL)
        self.timeline.append(new_token)
        self.last_label = new_token

    def push_direction(self, dir: Dir8, timestamp_sec: float):
        if self.last_direction and self.last_direction.value == dir:
            self.last_direction.extend_time(timestamp_sec)
            return
        new_token = Token(dir, timestamp_sec, TokenType.DIRECTION)
        self.timeline.append(new_token)
        self.last_direction = new_token

    def __str__(self):
        s = ''
        for i in self.timeline:
            s += f'{i.value},'
        return s
