"""Access the frozen receiver inputs and observation-only test streams."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V14 = HERE.parent/"JFI_V14_dual_direction_accommodation"
sys.dont_write_bytecode = True
sys.path.insert(0,str(V14))

from v14_inputs import (DEFAULT_BASELINE, MODELS, digest, fresh_streams, load_json,
                        load_prepared, setup, write_json)
from trusted_prediction import TrustedConditioning

MODEL_PATH = Path("common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json")
BASELINE_COMMIT = "3c145beb43beee9672300030b7a3c83350973e37"
BUDGETS = (.001,.005,.01,.02)
