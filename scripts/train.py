"""Command-line entry point for Go2 train; implementation lives in step_03_training."""

from src.locomotion_rl.go2_baseline.step_03_training.train import (
  TrainConfig as TrainConfig,
  launch_training as launch_training,
  main as main,
  run_train as run_train,
)


if __name__ == "__main__":
  main()
