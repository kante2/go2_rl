"""Command-line entry point for Go2 play; implementation lives in step_04_evaluation."""

import sys
from pathlib import Path

# Direct script execution adds scripts/, so also expose the repository package.
if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.locomotion_rl.go2_baseline.step_04_evaluation.play import (
  PlayConfig as PlayConfig,
  main as baseline_main,
  run_play as run_play,
)


def main():
  if len(sys.argv) > 1 and sys.argv[1].startswith("Unitree-Go2-QuietPaw-LongStride-"):
    from src.locomotion_rl.go2_quietpaw_long_stride.step_04_evaluation.play import main as long_stride_main
    long_stride_main()
  elif len(sys.argv) > 1 and sys.argv[1].startswith("Unitree-Go2-QuietPaw-"):
    from src.locomotion_rl.go2_quietpaw.step_04_evaluation.play import main as quietpaw_main
    quietpaw_main()
  else:
    baseline_main()


if __name__ == "__main__":
  main()
