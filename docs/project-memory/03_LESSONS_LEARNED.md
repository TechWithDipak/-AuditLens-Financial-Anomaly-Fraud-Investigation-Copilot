# 03 Lessons Learned

## 2026-11-15 | Path Handling with Spaces in Virtualenv Binaries
Learned: Unix shebang lines (`#!/...`) do not reliably parse paths containing spaces (e.g. `/machine learning/`).
Mistake: Executing `.venv/bin/streamlit` resulted in `bad interpreter: /Users/dipakkumar/project/machine: no such file or directory`.
Root cause: Operating system kernel splits shebang line on the first space.
Fix: Created shell wrapper scripts in `.venv/bin/` invoking `exec "$DIR/python" -m streamlit "$@"` and `exec "$DIR/python" -m pytest "$@"`.
Prevention rule: Whenever paths may contain spaces, use directory resolution and module execution (`python -m <module>`) rather than relying on absolute path shebangs.

## 2026-11-15 | Forward Algorithm Scaling for High-Volume Sequential Scoring
Learned: In high-volume sequential scoring (1.85 million records across 980 cardholders), log-space forward passes with repeated logsumexp and Python loops take ~1,370 seconds.
Mistake: Initially attempted calling log-space forward on sliding windows in Python loops.
Root cause: Python per-call overhead and repeated transcendental function calls (`np.log`, `np.exp`).
Fix: Implemented Rabiner (1989) scaled forward algorithm with floating-point matrix multiplication and cumulative rolling log-scaling sums. Reduced execution time to 3.2 seconds with identical log-likelihood output matching `hmmlearn` to within $10^{-5}$.
Prevention rule: For discrete HMM sequence scoring over large datasets, use scaled forward algorithms instead of unvectorized log-space windowing.

## 2026-11-15 | Parquet Column Coverage for Interactive Exploration
Learned: If precomputed test tables omit raw or intermediate columns (such as PCA coordinates or raw demographics), downstream interactive exploration can fail on missing keys.
Mistake: Initial `test_scored.parquet` contained only subset columns, causing interactive PCA button to fail with KeyError.
Root cause: Assumptions about which columns the UI needed for secondary interactive buttons.
Fix: Precomputed PCA coordinates directly in `test_scored.parquet` and included all base features.
Prevention rule: In data artifact generation, always include all columns referenced by UI interactive widgets.

## 2026-11-15 | Palette Centralization and WCAG Contrast Verification
Learned: Hardcoding hex values across multiple component files creates maintenance debt and makes theme synchronization error-prone.
Mistake: Earlier implementations had hex codes dispersed across `app.py`, `src/report.py`, and test files.
Root cause: Direct string styling without centralized constants.
Fix: Exported palette constants from `src/config.py` and `.streamlit/config.toml` only, interpolating them into UI and report styles. Added automated WCAG 2.1 relative luminance and contrast ratio calculations in `tests/test_compliance.py`.
Prevention rule: Define color tokens once at the configuration layer and enforce single-source-of-truth via static compliance tests.

## 2026-11-16 | Decoupling Heavy Training Dependencies from Inference and Serving
Learned: Top-level module imports of training-only packages (like `hmmlearn`) cause hard crashes at runtime if end users execute serving applications (`app.py`) in lightweight or production environments where training libraries are omitted.
Mistake: Unconditionally importing `from hmmlearn.hmm import CategoricalHMM` in `src/hmm.py` and pickling `self.model` inside `CustomerHMM`.
Root cause: Python evaluates module-level imports when `app.py` imports functions (`encode_discrete_symbols`), and standard `pickle` resolves class paths during unpickling.
Fix: Made `hmmlearn` an optional training-time import with try-except fallback. Cached transition, emission, and start probability matrices directly as numpy arrays on `CustomerHMM`, setting `self.model = None` during serialization (`__getstate__`). Added dummy mock module registration in `sys.modules` for legacy pickle compatibility and a fallback instantiation in `app.py` from `metrics.json`.
Prevention rule: For models evaluated with hand-written inference algorithms (Forward, Viterbi), store parameters purely in native numpy formats and keep training-time library imports strictly localized or conditional.
