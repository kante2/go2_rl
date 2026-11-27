"""사전학습 baseline 보행 체크포인트로 QPace 신경망을 초기화한다 (QPaceCurriculumCfg.init_checkpoint).

baseline: actor 입력 47 (각속도·중력·명령·위상·관절각·관절속도·직전 행동), critic 입력 74, critic 출력 1.
QPace:    같은 47(74)개 뒤에 목표 dB 조건 1개를 더 받는다. critic은 reward·cost 두 개의 ξ(s)ᵀw(조건) 구조.

- actor: 모든 층과 행동 표준편차를 복사한다. 조건 입력 열의 가중치는 0으로 두어 시작 시점에는 baseline과
  완전히 같은 행동을 한다. 조건 칸의 관측 정규화는 [0, 1] 균등분포(평균 0.5, 분산 1/12)로 둔다.
- critic: baseline critic의 정규화와 앞 3층(74→512→256→128)을 두 critic의 상태 인코더 ξ(s)에 복사한다.
  마지막 층(128→64)과 조건 가중치 w(조건)는 새로 시작한다.
- optimizer, λ(PID), 학습 iteration은 새로 시작한다.
"""

from pathlib import Path

import torch

_UNIFORM_MEAN, _UNIFORM_VAR = 0.5, 1.0 / 12.0
_CRITIC_HIDDEN = ("0", "2", "4")  # baseline critic mlp / QPace state_features의 Linear 층 인덱스


def _extend_normalizer(dst: dict, src: dict, prefix_dst: str, prefix_src: str, n_src: int):
  for key in ("_mean", "_var", "_std"):
    t = dst[f"{prefix_dst}{key}"].clone()
    t[..., :n_src] = src[f"{prefix_src}{key}"]
    fill = {"_mean": _UNIFORM_MEAN, "_var": _UNIFORM_VAR, "_std": _UNIFORM_VAR ** 0.5}[key]
    t[..., n_src:] = fill
    dst[f"{prefix_dst}{key}"] = t
  dst[f"{prefix_dst}count"] = src[f"{prefix_src}count"].clone()


@torch.no_grad()
def warm_start_from_baseline(actor, critic, path: str | Path) -> dict:
  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f"init_checkpoint not found: {path}")
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  src_a, src_c = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
  if "mlp.0.weight" not in src_a or "mlp.6.weight" not in src_c or src_c["mlp.6.weight"].shape[0] != 1:
    raise ValueError(f"{path} is not a baseline (single-critic) RSL-RL checkpoint.")

  # ---- actor
  dst_a = {k: v.clone() for k, v in actor.state_dict().items()}
  n_actor = src_a["mlp.0.weight"].shape[1]
  if dst_a["mlp.0.weight"].shape[1] != n_actor + 1:
    raise ValueError(f"Actor input mismatch: baseline {n_actor} + 1 condition vs QPace {dst_a['mlp.0.weight'].shape[1]}.")
  for key, value in src_a.items():
    if key.startswith("obs_normalizer.") or key == "mlp.0.weight":
      continue
    if key not in dst_a or dst_a[key].shape != value.shape:
      raise ValueError(f"Actor parameter {key} does not match the baseline.")
    dst_a[key] = value.clone()
  w = torch.zeros_like(dst_a["mlp.0.weight"])
  w[:, :n_actor] = src_a["mlp.0.weight"]  # 조건 열 = 0
  dst_a["mlp.0.weight"] = w
  _extend_normalizer(dst_a, src_a, "obs_normalizer.", "obs_normalizer.", n_actor)
  actor.load_state_dict(dst_a, strict=True)

  # ---- critics
  dst_c = {k: v.clone() for k, v in critic.state_dict().items()}
  n_critic = src_c["mlp.0.weight"].shape[1]
  for head in ("reward", "cost"):
    if dst_c[f"{head}.state_features.0.weight"].shape[1] != n_critic:
      raise ValueError(f"Critic state input mismatch: baseline {n_critic}.")
    for i in _CRITIC_HIDDEN:
      for kind in ("weight", "bias"):
        src = src_c[f"mlp.{i}.{kind}"]
        key = f"{head}.state_features.{i}.{kind}"
        if dst_c[key].shape != src.shape:
          raise ValueError(f"Critic layer {key} does not match the baseline.")
        dst_c[key] = src.clone()
    for kind in ("_mean", "_var", "_std"):
      dst_c[f"{head}.normalizer.{kind}"] = src_c[f"obs_normalizer.{kind}"].clone()
    dst_c[f"{head}.normalizer.count"] = src_c["obs_normalizer.count"].clone()
  critic.load_state_dict(dst_c, strict=True)
  return {"path": str(path), "iteration": ckpt.get("iter"), "actor_inputs": n_actor, "critic_inputs": n_critic}
