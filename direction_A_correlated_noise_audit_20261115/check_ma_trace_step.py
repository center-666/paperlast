"""Exact local algebra check, not an audit of the complete Ma2024 theorem."""
from pathlib import Path
import json
import sympy as sp

p = Path(__file__).resolve().parent
oi = sp.eye(2)
oj = sp.diag(3, sp.Rational(1, 10))
wi = 1 / sp.trace(oi)
wj = 1 / sp.trace(oj)
pi = wi / (wi + wj)
pj = wj / (wi + wj)
of = pi * oi + pj * oj
beta = 1
assert sp.trace(oj) / (1 + beta) <= sp.trace(oi) <= sp.trace(of)
assert sp.trace(of.inv()) > sp.trace(oi.inv())
result = {
    "purpose": "Ma2024 Eq14 first CI fusion with Eq11 trace gate; local counterexample to an inverse-trace inference only",
    "Omega_i": str(oi), "Omega_j": str(oj),
    "pi_i": str(pi), "pi_j": str(pj),
    "Omega_f": str(of), "beta": beta,
    "trace_Omega_i": str(sp.trace(oi)), "trace_Omega_f": str(sp.trace(of)),
    "trace_inverse_i": str(sp.trace(oi.inv())), "trace_inverse_f": str(sp.trace(of.inv())),
    "trace_gate_passes": True, "inverse_trace_increases": True,
    "full_dynamic_theorem_disproved": False,
}
(p / "ma_trace_step_exact.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
