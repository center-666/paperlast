"""Read only the frozen receiver models and pre-existing experiment replays."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V13 = HERE.parent / "JFI_V13_pre_alarm_risk_aware_accommodation"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V13))

from inputs import (DEFAULT_BASELINE, MODELS, V11, V12, digest, fresh_streams, joint_cases,  # noqa: F401,E402
                    legacy_cases, load_json, load_prepared, setup, write_json)

PRIMARY = ("A0_onset_replay", "A1_H_mismatch", "A1_C_mismatch")


def check_hashes(root, expected):
    for name, sha in expected.items():
        if digest(root/name) != sha:
            raise ValueError(f"Frozen artifact changed: {name}")
