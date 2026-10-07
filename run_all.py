"""
Run every analysis in order, then the invariant tests.

    python run_all.py

Equivalent to running, one after another:
    python baseline.py            # baseline (results/baseline/)
    python run_ablation.py           # feature ablation + random-seed sweep (results/ablation/)
    python run_drift_analysis.py     # drift quantification (results/drift/)
    python run_coefficients.py       # logistic-regression coefficients (results/coefficients/)
    python run_model_comparison.py   # six-model comparison (results/model_comparison/)
    python test_pipeline.py       # invariant tests
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["baseline.py", "run_ablation.py", "run_drift_analysis.py",
           "run_coefficients.py", "run_model_comparison.py", "test_pipeline.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n{'#' * 70}\n# {script}\n{'#' * 70}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
    print("\nAll analyses and tests completed.")
