"""
run_papermill.py – Chạy tất cả notebooks theo thứ tự.
Dùng nbconvert (không cần Jupyter kernel riêng).
Cách dùng: python scripts/run_papermill.py
"""
import subprocess
import sys
import os

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_association_mining.ipynb",
    "notebooks/03_clustering.ipynb",
    "notebooks/04_modeling.ipynb",
    "notebooks/04b_anomaly_mining.ipynb",
    "notebooks/05_evaluation_report.ipynb",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_with_nbconvert():
    print(f"[run] Working directory: {ROOT}\n")
    for nb_path in NOTEBOOKS:
        full_path = os.path.join(ROOT, nb_path)
        print(f"[nbconvert] Running: {nb_path} ...")
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.kernel_name=python3",
                full_path
            ],
            cwd=ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Done: {nb_path}\n")
        else:
            print(f"  ✗ Error in {nb_path}:")
            print(result.stderr[-800:])
            print()


def run_with_pipeline():
    """Fallback: chạy trực tiếp bằng run_pipeline.py nếu nbconvert lỗi."""
    print("[fallback] Chạy run_pipeline.py trực tiếp...")
    script = os.path.join(ROOT, "scripts", "run_pipeline.py")
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    try:
        import nbconvert  # noqa
        run_with_nbconvert()
    except ImportError:
        print("[warn] nbconvert không có. Chạy pipeline script trực tiếp...\n")
        run_with_pipeline()
