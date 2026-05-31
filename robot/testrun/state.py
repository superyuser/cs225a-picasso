from enum import Enum, auto


class State(Enum):
    GOING_INIT = auto()
    EXECUTING_PATH = auto()
    RETURNING_INIT = auto()
    DONE = auto()
