import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demo_runs_offline_and_matches_captured_output():
    result = subprocess.run([sys.executable, str(ROOT / "demo" / "run_demo.py")],
                            capture_output=True, text=True, timeout=60, check=True)
    captured = (ROOT / "demo" / "output.txt").read_text(encoding="utf-8")
    assert result.stdout.splitlines() == captured.splitlines()
    assert "done" in result.stdout
