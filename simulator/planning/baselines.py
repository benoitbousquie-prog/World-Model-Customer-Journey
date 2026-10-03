from __future__ import annotations
import numpy as np
from simulator.env.actions import Action

class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
    def act(self, s):
        return Action(int(self.rng.integers(0, 4)))

class AlwaysDiscountPolicy:
    def act(self, s):
        return Action.DISCOUNT

class MyopicPolicy:
    """Greedy 1-step policy using a provided world model predictor.
    Expects predictor(s,a)->(p_buy,p_churn) and immediate reward expectation.
    """
    def __init__(self, model_wrapper, gamma: float = 0.95):
        self.m = model_wrapper
        self.gamma = float(gamma)

    def act(self, s):
        best_a, best_v = Action.NONE, -1e9
        for a in [Action.NONE, Action.EMAIL, Action.RECO, Action.DISCOUNT]:
            v = self.m.expected_immediate_reward(s, a)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
