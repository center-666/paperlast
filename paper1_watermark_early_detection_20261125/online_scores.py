"""Numerical implementation of the registered causal parity-subspace score."""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from matched_glrt import build_sender_geometry

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def kernel():
    source = HERE / "projection.c"
    folder = Path(tempfile.gettempdir()) / ("paper1_projection_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    binary = folder / ("projection.dll" if os.name == "nt" else "projection.so")
    if not binary.exists():
        if os.name == "nt":
            finder = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
            root = subprocess.check_output([str(finder), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
            setup = Path(root) / "VC/Auxiliary/Build/vcvars64.bat"
            script = folder / "build.cmd"
            script.write_text(f'@call "{setup}" >nul\n@if errorlevel 1 exit /b 1\n@cl /nologo /std:c11 /O2 /LD "{source}" /link /OUT:"{binary}"\n')
            subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=folder, check=True)
        else:
            subprocess.run([shutil.which("cc") or "cc", "-O3", "-shared", "-fPIC", str(source), "-lm", "-o", str(binary)], check=True)
    lib = ct.CDLL(str(binary))
    scan = lib.scan
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    scan.argtypes = [ct.c_int]*8 + [ptr]*5
    scan.restype = None
    return scan


def operators(model):
    if hasattr(model, "_score_operators"):
        return model._score_operators
    assert np.linalg.matrix_rank(model.A) == 2
    operators_by_window = {}
    for width, startup in ((6, True), (7, True), (8, True), (8, False)):
        records = [build_sender_geometry(model, int(j), startup, width) for _, j in model.links]
        blocks = np.stack([r.whitening_map.reshape(r.rank, width, 2).transpose(1,0,2) for r in records])
        ha = np.stack([r.attack_state_maps[0] for r in records])
        operators_by_window[width,startup] = (np.ascontiguousarray(blocks), np.ascontiguousarray(ha))
    model._score_operators = operators_by_window
    return operators_by_window


def matched(packets, keys, ops, first=5):
    packets, keys = (np.ascontiguousarray(v, dtype=np.float64) for v in (packets, keys))
    paths, ticks, edges, m = packets.shape
    if m != 2 or keys.shape != (*packets.shape, 2):
        raise ValueError("Received report and key dimensions differ")
    result = np.zeros((paths, ticks, edges))
    for (width, startup), (blocks, ha) in ops.items():
        begin = max(first, width-1 if startup else 8)
        end = min(ticks, width if startup else ticks)
        if begin < end:
            kernel()(paths, ticks, edges, width, blocks.shape[2], begin, end, int(startup),
                     packets, keys, blocks, ha, result)
    return result
