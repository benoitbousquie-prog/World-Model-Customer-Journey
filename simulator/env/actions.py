from __future__ import annotations
from enum import IntEnum

class Action(IntEnum):
    NONE = 0
    EMAIL = 1
    RECO = 2
    DISCOUNT = 3

ACTION_NAMES = {
    Action.NONE: "none",
    Action.EMAIL: "email",
    Action.RECO: "reco",
    Action.DISCOUNT: "discount",
}

NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}
