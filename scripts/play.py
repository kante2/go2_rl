"""Command-line entry point for Go2 play; implementation lives in step_04_evaluation."""

from src.locomotion_rl.go2_baseline.step_04_evaluation.play import (
  PlayConfig as PlayConfig,
  main as baseline_main,
  run_play as run_play,
)


def main():
  import sys

  if len(sys.argv) > 1 and sys.argv[1].startswith("Unitree-Go2-QuietPaw-"):
    from src.locomotion_rl.go2_quietpaw.step_04_evaluation.play import main as quietpaw_main
    quietpaw_main()
  else:
    baseline_main()


if __name__ == "__main__":
  main()
