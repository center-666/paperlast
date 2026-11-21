"""Explicit opt-in launcher for the unmodified historical complete_v2 engine.

Default invocation prints the planned run and exits. It never starts a long
simulation on import. --execute opts in. The original costly arbitrary-attack
optimization is disabled unless separately requested with --include-arbitrary.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source/simulation_revised_detection_metrics_complete_v2.py'
EXPECTED='2f4694e6180680ccd83ecc4ae60cb587b76276625c9d34c5946cfeca2d3e8a4a'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='explicitly run historical main(), not a source-only inspection')
    parser.add_argument('--mc-runs',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,default=HERE/'legacy_outputs')
    parser.add_argument('--include-arbitrary',action='store_true',help='also run the costly original arbitrary-attack optimizer')
    args=parser.parse_args()
    if args.mc_runs<1:parser.error('--mc-runs must be positive')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('historical source hash mismatch')
    planned=dict(source=SOURCE.name,execute=args.execute,MC_RUNS=args.mc_runs,
                 SIM_OUTPUT_DIR=str(args.output_dir.resolve()),
                 run_arbitrary_attack_optimization=args.include_arbitrary,
                 estimator='historical stacked KF; not a common-G reproduction',
                 original_1000_run_provenance_confirmed=False)
    print(json.dumps(planned,indent=2))
    if not args.execute:return
    os.environ['MC_RUNS']=str(args.mc_runs)
    os.environ['SIM_OUTPUT_DIR']=str(args.output_dir.resolve())
    os.environ.setdefault('MPLBACKEND','Agg')
    spec=importlib.util.spec_from_file_location('historical_complete_v2',SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.Params.run_attack_optimization=args.include_arbitrary
    module.main()


if __name__=='__main__':main()
