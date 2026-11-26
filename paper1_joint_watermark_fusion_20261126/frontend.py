"""Four-dimensional parity construction with D2 constant/burst packet templates."""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.linalg import block_diag, cholesky, solve_triangular

HERE = Path(__file__).resolve().parent
ROTATIONS = np.array((
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
), dtype=float)


@lru_cache(None)
def kernel():
    source = HERE / "projection.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / ("joint_parity_" + digest)
    directory.mkdir(exist_ok=True)
    library = directory / ("projection.dll" if os.name == "nt" else "projection.so")
    if not library.exists():
        if os.name == "nt":
            finder = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
            root = subprocess.check_output([
                str(finder), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"], text=True).strip()
            setup = Path(root) / "VC/Auxiliary/Build/vcvars64.bat"
            command = directory / "build.cmd"
            command.write_text(
                f'@call "{setup}" >nul\n@if errorlevel 1 exit /b 1\n'
                f'@cl /nologo /std:c11 /O2 /LD "{source}" /link /OUT:"{library}"\n')
            subprocess.run(["cmd.exe", "/d", "/c", str(command)], cwd=directory, check=True)
        else:
            subprocess.run([shutil.which("cc") or "cc", "-O3", "-shared", "-fPIC",
                            str(source), "-lm", "-o", str(library)], check=True)
    function = ct.CDLL(str(library)).scan
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = [ct.c_int]*8+[pointer]*5
    function.restype = None
    return function


def geometry(model, sender, width):
    H = model.H[sender]
    A, Q, R = model.A, model.Q, model.R[sender]
    observability = np.vstack([H @ np.linalg.matrix_power(A, t) for t in range(width)])
    U, singular, _ = np.linalg.svd(observability, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-10))
    parity = U[:, rank:].T
    raw_covariance = np.zeros((2*width, 2*width))
    for a in range(width):
        for b in range(width):
            shared = np.zeros_like(Q)
            for process in range(min(a, b)):
                left = np.linalg.matrix_power(A, a-1-process)
                right = np.linalg.matrix_power(A, b-1-process)
                shared += left @ Q @ right.T
            block = H @ shared @ H.T
            if a == b:
                block += R
            raw_covariance[2*a:2*a+2, 2*b:2*b+2] = block
    covariance = parity @ raw_covariance @ parity.T
    lower = cholesky(covariance, lower=True)
    whitening = solve_triangular(lower, parity, lower=True)
    np.testing.assert_allclose(whitening @ observability, 0., atol=2e-12)
    np.testing.assert_allclose(whitening @ raw_covariance @ whitening.T,
                               np.eye(len(parity)), atol=2e-12)
    return {"O": observability, "raw_covariance": raw_covariance,
            "parity": parity, "whitening": whitening, "state_rank": rank,
            "support_rank": len(parity)}


def operators(model):
    result = {}
    for width in (5, 6, 7, 8):
        geometry_rows = [geometry(model, int(j), width) for _, j in model.links]
        blocks = np.stack([g["whitening"].reshape(g["support_rank"], width, 2).transpose(1, 0, 2)
                           for g in geometry_rows])
        if blocks.shape[2] > 14:
            raise ValueError("Joint parity support exceeds the inherited kernel's audited row bound.")
        result[width] = (np.ascontiguousarray(blocks), geometry_rows)
    return result


def keys(uniforms, no_repeat):
    state = np.floor(4*uniforms).astype(np.uint8)
    if no_repeat:
        for tick in range(1, uniforms.shape[1]):
            state[:, tick] = (state[:, tick-1]+1+np.floor(3*uniforms[:, tick]).astype(np.uint8)) % 4
    return ROTATIONS[state]


def template_activities(width):
    activities = [np.ones(width)]
    activities.extend(((np.arange(width)+phase) % 10 < 6).astype(float) for phase in range(10))
    return activities


def scores(packets, matrices, ops):
    packets = np.ascontiguousarray(packets, dtype=np.float64)
    matrices = np.ascontiguousarray(matrices, dtype=np.float64)
    count, ticks, edges, _ = packets.shape
    result = np.zeros((count, ticks, edges))
    for width, (blocks, _) in ops.items():
        begin = width if width < 8 else 8
        end = min(ticks, width+1) if width < 8 else ticks
        if begin >= end:
            continue
        for active in template_activities(width):
            maps = np.ascontiguousarray(
                np.broadcast_to(active[None, :, None, None]*np.eye(2), (edges, width, 2, 2)))
            temporary = np.zeros_like(result)
            kernel()(count, ticks, edges, width, blocks.shape[2], begin, end, 0,
                     packets, matrices, blocks, maps, temporary)
            result = np.maximum(result, temporary)
    return result


def independent_score(record, matrices, whitening):
    """Least-squares projection using direct SVD for every phase and suffix."""
    width = len(record)
    observation = whitening @ record.ravel()
    candidates = []
    for activity in template_activities(width):
        for start in range(width):
            design = np.zeros((2*width, 2))
            for t in range(start, width):
                design[2*t:2*t+2] = activity[t]*matrices[t]
            X = whitening @ design
            left, singular, _ = np.linalg.svd(X, full_matrices=False)
            bound = sum(np.linalg.norm(whitening[:, 2*t:2*t+2], "fro") *
                        np.linalg.norm(activity[t]*matrices[t], "fro")
                        for t in range(start, width))
            largest = singular[0] if len(singular) else 0.
            basis = left[:, singular > max(largest*1e-10, bound*1e-12)]
            candidates.append(float(np.sum((basis.T @ observation)**2)))
    return max(candidates)
