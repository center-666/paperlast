"""V1.1 scalar policy over the unchanged conditional observer and common-G update."""
from math import sqrt

import numpy as np

from weight_experiment import Weight


def suffix_gramians(feedback, horizon):
    """Penalty of the propagation of one posterior displacement to episode end."""
    identity = np.eye(feedback.shape[0])
    matrices = np.empty((horizon,) + identity.shape)
    suffix = identity
    for k in range(horizon - 1, -1, -1):
        matrices[k] = suffix
        suffix = identity + feedback.T @ suffix @ feedback
    return matrices


def nominal_limit(curvature, linear, budget):
    """Largest weight in [0,1] with curvature*a**2 - 2*linear*a <= budget."""
    if curvature - 2 * linear <= budget:
        return 1.0
    if curvature == 0:
        return budget / (-2 * linear)
    radius = sqrt(linear * linear + curvature * budget)
    if linear < 0:
        return budget / (radius - linear)
    return (radius + linear) / curvature


def conditional_weight(preview, posterior_mean, null_mean, direction, suffix, budget):
    squared = float(direction @ direction)
    if squared <= 1e-24:
        return Weight(0.0, 1.0, False, False, 0.0, float(budget))
    linear = float(direction @ (preview - posterior_mean))
    proposal = float(np.clip(linear / squared, 0.0, 1.0))
    weighted = suffix @ direction
    nominal_curvature = float(direction @ weighted)
    nominal_linear = float(weighted @ (preview - null_mean))
    cap = nominal_limit(nominal_curvature, nominal_linear, budget)
    alpha = min(proposal, cap)
    increment = alpha * (nominal_curvature * alpha - 2 * nominal_linear)
    return Weight(alpha, float(cap), False, cap < proposal - 1e-12,
                  float(increment), float(budget))


class TailBudgetPolicy:
    """Minimize one-step working posterior risk subject to a true H0 budget."""

    def __init__(self, prepared):
        settings = prepared.settings
        receiver = settings.receiver
        self.gain = prepared.model.G[receiver]
        self.suffix = suffix_gramians(prepared.model.F(receiver), settings.horizon)
        self.budget = settings.nominal_one_step_allowance * prepared.nominal_mse()

    def __call__(self, k, preview, evidence):
        return conditional_weight(preview, evidence.mixture_state_mean, evidence.nominal_state_mean,
                                  self.gain @ evidence.delta, self.suffix[k], self.budget[k])
