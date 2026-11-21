"""Offline geometry audit of T3 on fixed, synthetic estimate vectors.

No traffic, attack replay, fitting or online use of true state. Standard library only.
"""

import csv
import math
from pathlib import Path


# W = diag(4, 1/4); the second coordinate is the mode tracked separately.
SCALE = (2.0, 0.5)
BUDGETS = (0.0, 0.04, 0.25, 1.0, 4.0, 25.0)
# Each row is (name, truth, nominal center, raw output, pre-projection candidate).
SCENARIOS = (
    ("center_on_raw", (0.0, 0.0), (0.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
    ("room_for_recovery", (0.0, 0.0), (0.0, 0.5), (0.0, 3.0), (0.0, 0.25)),
    ("wrong_direction", (0.0, 0.0), (0.0, 4.0), (0.0, 2.0), (0.0, 7.0)),
    ("mode_and_state_differ", (1.0, 0.0), (0.0, 2.0), (-0.5, 2.0), (0.2, 0.0)),
)


def distance(x, y):
    return math.hypot(*((a - b) * s for a, b, s in zip(x, y, SCALE)))


def project(candidate, center, radius):
    d = distance(candidate, center)
    t = min(1.0, radius / d) if d else 1.0
    return tuple(c + t * (x - c) for x, c in zip(candidate, center))


def score(scenario, budget):
    name, truth, center, raw, candidate = scenario
    radius = math.hypot(distance(raw, center), math.sqrt(budget))
    out = project(candidate, center, radius)  # Does not read truth.
    floor_w = max(0.0, distance(truth, center) - radius) ** 2
    floor_mode = max(0.0, abs(truth[1] - center[1]) - radius / SCALE[1]) ** 2
    row = {"scenario": name, "budget": budget, "radius_w": radius}
    for label, z in (("raw", raw), ("candidate", candidate), ("final", out)):
        row[f"{label}_loss_w"] = distance(z, truth) ** 2
        row[f"{label}_loss_mode"] = (z[1] - truth[1]) ** 2
    for suffix, floor in (("w", floor_w), ("mode", floor_mode)):
        row[f"floor_{suffix}"] = floor
        row[f"headroom_{suffix}"] = row[f"raw_loss_{suffix}"] - floor
        row[f"remaining_{suffix}"] = row[f"final_loss_{suffix}"] - floor
        row[f"gain_{suffix}"] = row[f"raw_loss_{suffix}"] - row[f"final_loss_{suffix}"]
    row["projection_displacement_w"] = distance(out, candidate)
    row["candidate_clipped"] = distance(candidate, center) > radius
    assert distance(out, center) ** 2 <= distance(raw, center) ** 2 + budget + 1e-12
    return row


def run():
    results = []
    for scenario in SCENARIOS:
        previous = None
        for budget in BUDGETS:
            row = score(scenario, budget)
            assert row["headroom_w"] >= -1e-12 and row["headroom_mode"] >= -1e-12
            assert row["remaining_w"] >= -1e-12 and row["remaining_mode"] >= -1e-12
            if previous is not None:
                assert row["floor_w"] <= previous["floor_w"] + 1e-12
                assert row["floor_mode"] <= previous["floor_mode"] + 1e-12
            previous = row
            results.append(row)

    # Check against a separate direct search of the ellipse boundary plus its
    # interior witness (truth for the state metric, matching y for the mode).
    angles = [math.tau * k / 4096 for k in range(4096)]
    for row in results:
        _, truth, center, _, _ = next(s for s in SCENARIOS if s[0] == row["scenario"])
        r = row["radius_w"]
        points = [(center[0] + r * math.cos(a) / SCALE[0],
                   center[1] + r * math.sin(a) / SCALE[1]) for a in angles]
        if distance(truth, center) <= r:
            points.append(truth)
        mode_match = (center[0], truth[1])
        if distance(mode_match, center) <= r:
            points.append(mode_match)
        assert abs(min(distance(z, truth) ** 2 for z in points) - row["floor_w"]) < 1e-4
        assert abs(min((z[1] - truth[1]) ** 2 for z in points) - row["floor_mode"]) < 1e-10
    return results


if __name__ == "__main__":
    rows = run()
    out_file = Path(__file__).with_name("synthetic_results.csv")
    with out_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} fixed synthetic cases; wrote {out_file.name}")
