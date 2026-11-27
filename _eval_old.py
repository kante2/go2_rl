from pathlib import Path
from src.locomotion_rl.go2_qpace.step_04_evaluation.evaluate import EvaluationConfig, evaluate
run = Path("logs/rsl_rl/go2_qpace/2026-11-27_06-17-07_qpace_34997_full")
for it in (0, 1000, 1500, 2000, 2300, 2500):
  print(f"===== model_{it}", flush=True)
  evaluate(EvaluationConfig(checkpoint_file=run / f"model_{it}.pt", target_dbs=(-58., -55., -52., -48.),
                            velocities=(0.3, 0.5, 0.7), num_envs=64, steps=1000,
                            output=Path(f"logs/go2_qpace_eval_old/model_{it}.csv")))
