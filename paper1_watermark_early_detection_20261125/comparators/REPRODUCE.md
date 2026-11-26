# Reproduction

Start in an independent checkout of the results branch. Work in paper1_watermark_early_detection_20261125/comparators. Physical baseline da7a6c7d92b4bbf3514e86dbe51cca15e84660a5, registered implementation c3ac9c3, confirm lock 60fc7946f98794be18caf889846a061d66a271fc. The confirmation includes all six native encodings, two energy-matched encodings and six template-shift arms, 22 combinations in total.

Windows environment: Python 3.11.18, NumPy 2.4.6, SciPy 1.17.1, MSVC 2022 Build Tools. On POSIX the upstream kernel uses cc. Exact reference hashes in the physical backend can be sensitive to text line endings; use a Git checkout preserving the upstream attributes and LF source text. A locally built, source-hashed DLL is cached in the temporary directory. The deterministic second-moment comparison also reads the cb3ad287 commit object from this repository.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
python -B ../verify_archives.py
python -B verify_results.py
python -B tables.py
python -B orthogonal_readback.py
python -B moments.py
```

For W0 replay and a full prospective reproduction, begin at c3ac9c3 in a separate checkout without the later generated dev/confirm outputs. From that revision's comparators directory run the following. audit.py verifies every path of the two chosen historical cells. It compares the available early and path score maxima, censored first alarms and losses with archived arrays, and compares newly generated full scores and gates across two implementations. The archived full scores and gates do not exist. The old verifier regenerates two historical reading tables; the frozen input arrays are read only.

```powershell
python -B experiment.py fit
python -B audit.py
python -B experiment.py dev --workers 8
python -B analyze.py dev
python -B experiment.py confirm --workers 8
python -B analyze.py confirm
```

The numerical inputs in the new lock should reproduce. A new run includes the measured timings and hence has its own development manifest digest. The shipped lock binds to the shipped archive. The split, scene, population and batch uniquely determine all physical/key streams. Population 0 is calibration, 1 is independent health and 2 is evaluation. Each NPZ path_id is local to its file. Arrays for all arms are paired on that row. Scene 0..5 equals 2 times the index of (0,−45,+45) plus the index of onset (50,65). The 60-valued edge delay means censored. Loss columns are post20 affected/network, disturbance affected/network, and recovery affected/network, in that order. For healthy paths only the first pair applies.

Main tables and per-scene CSVs include fractional probabilities, not percentages. contrast lower/upper are two-sided, scenario-stratified path bootstrap intervals. Absolute primary FWER values use independent healthy holdouts and marginal path-level calibration. Resource rows include independent health transmit energy and schedule construction time. Absolute.csv/contrasts.csv use the post20 columns over the *evaluated* paths, health.csv uses separate healthy paths. Timing columns sum worker wall seconds and depend on batching and caching. Shared healthy plugin and globally proportional matched models share their healthy calculations. Each filter replay includes an internal legacy plugin pass. The confirmation wall clock was 271.65 seconds on the recorded host; hardware timing does not establish per-device implementation cost.

[Pre-registration](PROTOCOL.md) · [Fit of the independent energy multipliers](ENERGY_LOCK.json) · [Confirmation run manifest](confirm/RUN.json) · [Confirmation path readback](VERIFICATION.json)
