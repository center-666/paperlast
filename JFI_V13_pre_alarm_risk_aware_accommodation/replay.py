"""Receiver recursion, paired displacement checks, and closed-loop metrics."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, t

from inputs import digest  # noqa: F401
from analytical_weight import suffix_gramians
from nominal_audit import cap
from batched_observer import BatchedObserver

BASE = ("V1a", "V1.2a", "fixed-0", "fixed-0.25", "fixed-0.5", "fixed-0.75", "two-mode", "conditional-MSE")


@dataclass(frozen=True)
class Observations:
    """The only packets visible to a receiver run."""
    trusted: np.ndarray
    target: np.ndarray
    decoder: np.ndarray

    def __len__(self):
        return len(self.target)


def play(prepared, records, methods=BASE, risk_factory=None):
    obs = Observations(records.trusted, records.target, records.decoder)
    observer = BatchedObserver(prepared)
    model, s = prepared.model, prepared.settings
    f, g, K = model.F(s.receiver), model.G[s.receiver], model.K[s.receiver]
    horizon, m, n = s.horizon, prepared.m, prepared.n
    count, kinds = len(obs), len(methods)
    scores = np.full((count, kinds, horizon), np.nan)
    hist = np.zeros((count, kinds, n))
    posts = np.empty((count, kinds, horizon, n))
    alpha = np.empty((count, kinds, horizon))
    evidence = np.empty((count, horizon, m))
    null_score = np.empty_like(alpha)
    log_bf = np.empty((count, horizon))
    active = np.empty_like(log_bf)
    alarmed = np.zeros(count, dtype=bool)
    credit = np.zeros(count)
    suffixes = suffix_gramians(f, horizon)
    allocations = s.nominal_one_step_allowance * prepared.nominal_mse()
    predictor = risk_factory(prepared, methods, count) if risk_factory is not None else None

    for k in range(horizon):
        frame = observer.at(obs, k, risk=predictor is not None)
        evidence[:, k] = frame.delta
        log_bf[:, k] = frame.log_bf
        active[:, k] = frame.probability_active
        alarmed |= frame.log_bf >= -np.log(s.event_false_alarm)
        trusted = obs.trusted[:, k]
        raw = trusted[:, :m] @ K.T + (trusted[:, m:] + obs.target[:, k]) @ g.T
        preview = hist @ f.T + raw[:, None, :]
        d = frame.delta @ g.T
        mse_direction = preview-frame.mean_working[:, None, :]
        length2 = np.sum(d*d, axis=1)
        inner = np.einsum("nm,npm->np", d, mse_direction)
        proposal = np.divide(inner, length2[:, None], out=np.zeros_like(inner),
                             where=length2[:, None]>1e-24).clip(0., 1.)
        out = np.ones((count, kinds))
        for j, method in enumerate(methods):
            if method.startswith("fixed-"):
                out[:, j] = float(method.removeprefix("fixed-"))
            elif method == "two-mode":
                out[:, j] = np.where(~alarmed & (frame.probability_active<.35), .5, 1.)
            elif method == "conditional-MSE":
                out[:, j] = proposal[:, j]
            elif method == "V1.2a":
                sd = d @ suffixes[k]
                nom = preview[:, j]-frame.mean_null
                curve, line = np.sum(sd*d, axis=1), np.sum(sd*nom, axis=1)
                available = allocations[k]+credit
                chosen = np.minimum(proposal[:, j], cap(curve, line, available))
                charge = chosen*(curve*chosen-2*line)
                if np.max(charge-available) > 2e-11:
                    raise ArithmeticError("V1.2a budget exceeded.")
                credit = np.maximum(0., available-charge)
                out[:, j] = chosen
        if predictor is not None:
            out, p = predictor.choose(frame, k, mse_direction, d, out, ~alarmed, suffixes[k])
            scores[:, :, k] = p
        hist = preview-out[..., None]*d[:, None, :]
        alpha[:, :, k] = out
        posts[:, :, k] = hist
        null_score[:, :, k] = np.sum((hist-frame.mean_null[:, None, :])**2, axis=-1)
    return {"post": posts, "alpha": alpha, "delta": evidence, "log_bf": log_bf, "active": active,
            "predicted": scores, "null_score": null_score}


def run_batches(prepared, streams, methods=BASE, risk_factory=None, batch=128):
    runs = [play(prepared, streams.take(slice(i, i+batch)), methods, risk_factory)
            for i in range(0, len(streams), batch)]
    return {field: np.concatenate([part[field] for part in runs]) for field in runs[0]}


def first(mask):
    horizon = mask.shape[-1]
    return np.where(mask, np.arange(horizon), horizon).min(axis=-1)


def interval(values):
    a = np.asarray(values)
    average = float(a.mean())
    err = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return [average, average-err, average+err]


def paired_risk(new, base, tail=.025):
    n = len(new)
    worse = int(np.sum(new & ~base))
    better = int(np.sum(base & ~new))

    def upper(count):
        return float(beta.ppf(1-tail/2, count+1, n-count)) if count<n else 1.

    def lower(count):
        return float(beta.ppf(tail/2, count, n-count+1)) if count else 0.

    return {"difference": (worse-better)/n, "only_new": worse, "only_reference": better,
            "exact_interval": [lower(worse)-upper(better), upper(worse)-lower(better)],
            "tail_per_sign": tail}


def summarize(prepared, data, clean, streams, methods, danger_squared, tail=.025):
    """Truth is read only here, after both policy paths completed."""
    pred = data["post"]
    corr = data["alpha"][..., None]*data["delta"][:, None, :, :]
    corr0 = clean["alpha"][..., None]*clean["delta"][:, None, :, :]
    health_sq = np.sum((clean["post"]-streams.state[:, None])**2, axis=-1)
    err = np.sum((pred-streams.state[:, None])**2, axis=-1)
    bias = pred-clean["post"]
    f, g = prepared.model.F(prepared.settings.receiver), prepared.model.G[prepared.settings.receiver]
    prev = np.concatenate((np.zeros_like(bias[:, :, :1]), bias[:, :, :-1]), axis=2)
    local = streams.offset[:, None]-corr+corr0
    np.testing.assert_allclose(bias, prev @ f.T + local @ g.T, rtol=3e-11, atol=6e-12)
    np.testing.assert_allclose(err, health_sq+np.sum(bias*bias, axis=-1)+
                               2*np.sum((clean["post"]-streams.state[:, None])*bias, axis=-1),
                               atol=2e-11)
    offset = np.sum(bias*bias, axis=-1)
    T = pred.shape[2]
    first_alarm = first(data["log_bf"]>=-np.log(prepared.settings.event_false_alarm))
    first_danger = first(offset>=danger_squared)
    event = first_danger < first_alarm[:, None]
    ticks = np.arange(T)[None]
    before = ticks<streams.start[:, None]
    first_packet = ticks==streams.start[:, None]
    support = (ticks>=streams.start[:, None]) & (ticks<streams.stop[:, None])
    late = support & ~first_packet
    after = ticks>=streams.stop[:, None]
    if np.any(offset*before[:, None, :] > 1e-23):
        raise AssertionError("Receiver paths differ before the recorded onset.")
    mse = np.divide(np.sum(err*support[:, None, :], axis=2), np.maximum(support.sum(axis=1)[:, None], 1))
    recovery = np.divide(np.sum(err*after[:, None, :], axis=2), np.maximum(after.sum(axis=1)[:, None], 1))
    healthy_mse = health_sq.mean(axis=2)
    results = {}
    spans = (("pre", before), ("first", first_packet), ("active_later", late), ("after", after))
    true_residual = streams.offset[:, None]-corr
    for j, key in enumerate(methods):
        phase = {}
        for label, mask in spans:
            if np.any(mask):
                phase[label] = {"n": int(mask.sum()),
                                "mse": float(err[:, j][mask].mean()),
                                "mean_alpha": float(data["alpha"][:, j][mask].mean()),
                                "alpha_quantile": np.quantile(data["alpha"][:, j][mask], [.1,.5,.9]).tolist(),
                                "uncompensated_direction_sq": float(np.sum(true_residual[:, j]**2, axis=-1)[mask].mean()),
                                "mean_correction_sq": float(np.sum(corr[:, j]**2, axis=-1)[mask].mean()),
                                "correction_nonzero_fraction": float(np.mean(np.linalg.norm(corr[:, j], axis=-1)[mask]>1e-6)),
                                "residual_displacement_sq": float(offset[:, j][mask].mean())}
        src, dst = mse[:, 0], mse[:, j]
        first_actual = np.any((offset[:, j]>=danger_squared)&first_packet, axis=1)
        first_base = np.any((offset[:, 0]>=danger_squared)&first_packet, axis=1)
        results[key] = {"healthy_mse": interval(healthy_mse[:, j]),
                        "active_mse": interval(dst) if support.any() else None,
                        "mse_difference": interval(dst-src) if support.any() else None,
                        "mse_relative": float(dst.mean()/src.mean()-1) if support.any() else None,
                        "after_mse": interval(recovery[:, j]) if after.any() else None,
                        "alarm_before_window_end": int(np.sum(first_alarm<T)),
                        "first_packet_danger": int(first_actual.sum()),
                        "risk_before_alarm": int(event[:, j].sum()),
                        "paired_first": paired_risk(first_actual, first_base, tail),
                        "paired_risk": paired_risk(event[:, j], event[:, 0], tail),
                        "phases": phase,
                        "peak_displacement": interval(offset[:, j].max(axis=1))}
    details = {"mse": mse, "tail_mse": recovery, "healthy_mse": healthy_mse, "offset_sq": offset,
               "max_offset_sq": offset.max(axis=2), "first_alarm": first_alarm, "first_danger": first_danger,
               "risk": event, "first_packet": np.any((offset>=danger_squared)&first_packet[:, None], axis=2),
               "alpha": data["alpha"], "risk_score": data["predicted"], "p_active": data["active"],
               "start": streams.start, "stop": streams.stop, "after_mask": after}
    return results, details
