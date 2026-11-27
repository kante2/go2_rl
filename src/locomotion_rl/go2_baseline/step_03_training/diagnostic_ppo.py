"""기본 RSL-RL PPO에 GAE/advantage/KL 진단 값을 tensorboard 로그로 추가한다.

학습 방식은 PPO와 완전히 같다. update()가 돌려주는 dict는 Runner가 Loss/<key>로
기록하므로, 진단 값도 같은 dict에 넣어 Loss/advantages_mean 등의 이름으로 남긴다.
다중 GPU에서는 rank 0의 rollout 통계만 기록된다.
"""

import torch
from rsl_rl.algorithms import PPO


class DiagnosticPPO(PPO):
  def compute_returns(self, obs):
    super().compute_returns(obs)
    st = self.storage
    # super()가 st.advantages를 표준화하므로 원래 척도의 GAE는 returns - values로 다시 구한다.
    advantages = st.returns - st.values
    returns_var = st.returns.var()
    self.rollout_metrics = {
      "returns_mean": st.returns.mean().item(),  # GAE return 목표 R_t = A_t + V(s_t).
      "returns_std": st.returns.std().item(),
      "values_mean": st.values.mean().item(),  # rollout 중 critic 예측 V(s_t).
      "advantages_mean": advantages.mean().item(),
      "advantages_std": advantages.std().item(),
      # 1에 가까울수록 critic이 return을 잘 설명한다. 0 이하이면 critic이 상수 예측보다 못하다.
      "explained_variance": (1.0 - advantages.var() / (returns_var + 1e-8)).item(),
    }

  def update(self):
    # adaptive schedule이 미니배치마다 계산하는 KL을 가로채 평균만 기록한다.
    kls = []
    get_kl = self.actor.get_kl_divergence

    def recording_kl(old_params, new_params):
      kl = get_kl(old_params, new_params)
      kls.append(kl.mean())
      return kl

    self.actor.get_kl_divergence = recording_kl
    try:
      loss_dict = super().update()
    finally:
      del self.actor.get_kl_divergence  # 인스턴스 속성을 지워 클래스 메서드로 되돌린다.
    loss_dict.update(getattr(self, "rollout_metrics", {}))
    if kls:
      loss_dict["kl"] = torch.stack(kls).mean().item()
    return loss_dict
