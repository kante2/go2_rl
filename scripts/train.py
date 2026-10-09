"""Command-line entry point for Go2 train; implementation lives in step_03_training."""

from src.locomotion_rl.go2_baseline.step_03_training.train import (
  TrainConfig as TrainConfig,
  launch_training as launch_training,
  main as baseline_main,
  run_train as run_train,
)


def main():
  import sys

  if len(sys.argv) > 1 and sys.argv[1].startswith("Unitree-Go2-QuietPaw-"):
    from src.locomotion_rl.go2_quietpaw.step_03_training.train import main as quietpaw_main
    quietpaw_main()
  else:
    baseline_main()


if __name__ == "__main__":
  main()
