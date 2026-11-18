"""Read frozen nominal models and receiver-only utilities."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/"JFI_secure_estimation_accuracy_safety_frontier"
sys.dont_write_bytecode = True
sys.path.insert(0,str(PREVIOUS))

from frozen_inputs import (DEFAULT_BASELINE, MODELS, MODEL_PATH, TrustedConditioning, V14, digest, load_json,
                           load_prepared, setup, write_json)
from frontier import base_rational, covariances, rounded_exact

BASE_COMMIT = "afcab9e4fa8728fab01ab1edc8420079cf474c7a"


def validate_spec(settings,spec):
    assert settings.horizon==32
    assert settings.event_false_alarm==.05
    assert settings.coded_energy_reference==10
    assert spec["danger_squared"]==.1
    assert spec["watermark"]=={"lambda":.5,"q":.5}
