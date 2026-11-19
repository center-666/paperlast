"""Finite-horizon receiver events and paired state-error measurements."""

from dataclasses import dataclass

import numpy as np


DANGER_SQUARED = 0.1
BF_THRESHOLD = 20.0


def first_crossing(flags):
    """A time equal to the number of ticks denotes right censoring."""
    flags = np.asarray(flags)
    if flags.dtype != np.bool_ or flags.ndim != 2 or not flags.shape[1]:
        raise ValueError("Expected a Boolean paths-by-ticks array.")
    horizon = flags.shape[1]
    return np.min(np.where(flags, np.arange(horizon), horizon), axis=1)


@dataclass(frozen=True)
class Events:
    danger: np.ndarray
    alarm: np.ndarray
    onset: np.ndarray
    stop: np.ndarray
    horizon: int

    @property
    def observed(self):
        return self.danger < self.horizon

    @property
    def before(self):
        return self.observed & (self.danger < self.alarm)

    @property
    def first_packet(self):
        return self.observed & (self.danger == self.onset)

    def counts(self):
        observed_alarm = self.alarm < self.horizon
        both = self.observed & observed_alarm
        flags = {
            "any_danger": self.observed,
            "before_alarm": self.before,
            "same_tick": both & (self.danger == self.alarm),
            "after_alarm": both & (self.danger > self.alarm),
            "first_packet": self.first_packet,
            "first_packet_before_alarm": self.first_packet & self.before,
            "danger_alarm_censored": self.observed & ~observed_alarm,
            "alarm_danger_censored": observed_alarm & ~self.observed,
            "both_censored": ~self.observed & ~observed_alarm,
            "alarm_observed": observed_alarm,
            "alarm_before_onset": observed_alarm & (self.alarm < self.onset),
            "first_danger_at_or_after_stop": self.observed & (self.danger >= self.stop),
        }
        return {name: int(flag.sum()) for name, flag in flags.items()}


def events_from_times(danger, alarm, onset, stop, horizon):
    times = tuple(np.asarray(values) for values in (danger, alarm, onset, stop))
    shape = times[0].shape
    if len(shape) != 1 or not shape[0] or horizon < 1:
        raise ValueError("At least one finite-horizon path is required.")
    for values in times:
        if (values.shape != shape or not np.issubdtype(values.dtype, np.integer)
                or np.any((values < 0) | (values > horizon))):
            raise ValueError("Event and support times must be aligned integer indices.")
    danger, alarm, onset, stop = times
    if np.any(onset > stop) or np.any(danger < onset):
        raise ValueError("Invalid support or a difference before the recorded onset.")
    return Events(danger, alarm, onset, stop, horizon)


def masked_mean(values, mask):
    counts = mask.sum(axis=1)
    return np.divide(
        np.where(mask, values, 0.0).sum(axis=1),
        counts,
        out=np.full(len(values), np.nan),
        where=counts > 0,
    )


@dataclass(frozen=True)
class PathMetrics:
    events: Events
    healthy_mse: np.ndarray
    support_mse: np.ndarray
    tail_mse: np.ndarray
    full_mse: np.ndarray | None = None


def score_estimates(attacked_post, healthy_post, truth, log_bf, onset, stop):
    """Score two completed closed loops using the same saved physical truth."""
    attacked_post, healthy_post, truth = map(np.asarray, (attacked_post, healthy_post, truth))
    if (attacked_post.ndim != 3 or attacked_post.shape != healthy_post.shape
            or attacked_post.shape != truth.shape or not attacked_post.shape[-1]):
        raise ValueError("Expected aligned paths-by-ticks-by-state estimates.")
    if not all(np.all(np.isfinite(x)) for x in (attacked_post, healthy_post, truth)):
        raise ValueError("Estimates and truth must be finite.")
    log_bf = np.asarray(log_bf)
    if log_bf.shape != attacked_post.shape[:2] or np.any(np.isnan(log_bf)):
        raise ValueError("The original BF must have one observed score per tick and path.")
    paired_squared = np.sum((attacked_post - healthy_post) ** 2, axis=-1)
    horizon = paired_squared.shape[1]
    events = events_from_times(
        first_crossing(paired_squared >= DANGER_SQUARED),
        first_crossing(log_bf >= np.log(BF_THRESHOLD)),
        onset, stop, horizon,
    )
    ticks = np.arange(horizon)[None, :]
    support = (ticks >= events.onset[:, None]) & (ticks < events.stop[:, None])
    tail = ticks >= events.stop[:, None]
    squared_errors = np.sum((attacked_post - truth) ** 2, axis=-1)
    return PathMetrics(
        events,
        np.sum((healthy_post - truth) ** 2, axis=-1).mean(axis=1),
        masked_mean(squared_errors, support),
        masked_mean(squared_errors, tail),
        squared_errors.mean(axis=1),
    )
