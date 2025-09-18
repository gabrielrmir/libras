import time
from enum import Enum

class Direction(Enum):
    NONE = 0
    RIGHT = 1
    TOP_RIGHT = 2
    TOP = 3
    TOP_LEFT = 4
    LEFT = 5
    BOTTOM_LEFT = 6
    BOTTOM = 7
    BOTTOM_RIGHT = 8

# Representação de um frame
# Um state contém as seguintes informações:
# - Gesto
# - Direção
# - Rotação
# - Tempo em milisegundos

# Cada frame é feita a captura desses dados,
# caso seja diferente do último estado no array de estados,
# este estado é acrescentado ao array

class State():
    def __init__(self):
        self.timestamp_ms = int(time.time()*1000)
        self.label = ''
        self.world_pos = (0,0)
        self.direction = Direction.NONE

    def __eq__(self, other):
        return self.label == other.label and \
            self.direction == other.direction

class StateHandler():
    def __init__(self):
        self.states = []
        self.last_position = (0,0)
        self._start = True
        self.max_states = 10
    
    # Extrair informações dos landmarks e 
    # criar estado a partir dessas informações
    # já se assume que os landmarks já foram classificados
    def Update(self, landmarks, label):
        # if not gesture: return
        # if len(self.states) > 0 and new_state == self.states[-1]: return
        # self.states.append(new_state)
        # self.last_position = landmarks.root_position
        # TODO: limitar tamanho do array para número máximo de elementos
        pass
