from __future__ import annotations

import numpy as np
from simulator.env.actions import Action

class RandomizedLoggingPolicy:
    def __init__(self, action_probs: dict[str, float], seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.actions = [Action.NONE, Action.EMAIL, Action.RECO, Action.DISCOUNT]
        self.p = np.array([action_probs["none"], action_probs["email"], action_probs["reco"], action_probs["discount"]], dtype=float)
        self.p = self.p / self.p.sum()

    def act(self, s):
        a = self.rng.choice(self.actions, p=self.p)
        return Action(int(a))

class ConfoundedLoggingPolicy:
    """Simple business-as-usual confounding:
    - discount more likely when P is low
    - email more likely when E is high
    - none more likely when F is high
    """
    def __init__(self, discount_when_low_P: float, email_when_high_E: float, none_when_high_F: float, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.discount_when_low_P = float(discount_when_low_P)
        self.email_when_high_E = float(email_when_high_E)
        self.none_when_high_F = float(none_when_high_F)

    def act(self, s):
        E, F, P = float(s[0]), float(s[1]), float(s[2])

        # base logits (start near uniform, then tilt)
        logits = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)  # none, email, reco, discount

        # if fatigue high, prefer none
        if F > 0.7:
            logits[0] += self.none_when_high_F

        # if engagement high, prefer email and reco
        if E > 0.6:
            logits[1] += self.email_when_high_E
            logits[2] += 0.3 * self.email_when_high_E

        # if propensity low, prefer discount
        if P < 0.3:
            logits[3] += self.discount_when_low_P

        # softmax
        ex = np.exp(logits - logits.max())
        p = ex / ex.sum()
        a_idx = int(self.rng.choice([0,1,2,3], p=p))
        return Action(a_idx)
