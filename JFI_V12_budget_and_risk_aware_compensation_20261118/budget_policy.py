"""V1.2a receiver weight with causal carryover of nominal risk allowance."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V11 = HERE.parent / "JFI_V11_soft_weight_reconstruction_20261118"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V11))

from analytical_weight import conditional_weight, suffix_gramians  # noqa: E402


@dataclass(frozen=True)
class BudgetDecision:
    alpha: float
    cap: float
    clipped_by_budget: bool
    credit_before: float
    allocation: float
    available: float
    nominal_charge: float
    credit_after: float


class CarryBudgetPolicy:
    """One instance per episode. Uses no data beyond the current evidence."""

    def __init__(self, prepared):
        settings = prepared.settings
        receiver = settings.receiver
        self.gain = prepared.model.G[receiver]
        self.suffixes = suffix_gramians(prepared.model.F(receiver), settings.horizon)
        self.stage_budget = settings.nominal_one_step_allowance * prepared.nominal_mse()
        self.credit = 0.0
        self.next_timestamp = 0

    def step(self, k, preview, evidence):
        if k != self.next_timestamp or k != evidence.timestamp or k >= len(self.stage_budget):
            raise ValueError("Expected a single update for the next registered timestamp.")
        opening = self.credit
        allocation = float(self.stage_budget[k])
        available = opening + allocation
        decision = conditional_weight(preview, evidence.mixture_state_mean,
                                      evidence.nominal_state_mean, self.gain @ evidence.delta,
                                      self.suffixes[k], available)
        charge = decision.tail_nominal_increment
        credit = available - charge
        if credit < -8 * np.finfo(float).eps * max(1.0, available):
            raise ArithmeticError("Nominal budget root exceeded its available allowance.")
        self.credit = max(0.0, credit)
        self.next_timestamp += 1
        return BudgetDecision(decision.alpha, decision.cap, decision.cap_binds, opening,
                              allocation, available, charge, self.credit)
