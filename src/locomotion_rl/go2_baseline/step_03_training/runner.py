import os

# 기본 PPO Runner를 확장해 .pt 체크포인트와 추론용 policy.onnx를 함께 저장한다.

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper  # RSL-RL 인터페이스로 감싼 환경; 실제 mjlab 환경은 env.unwrapped.

  def save(self, path: str, infos=None):
    super().save(path, infos)  # 모델·옵티마이저·반복 번호와 부가 상태(infos)를 .pt로 저장한다.
    policy_path = path.split("model")[0]  # model_*.pt 앞의 디렉터리 경로를 얻는 현재 구현.
    filename = "policy.onnx"  # 같은 실행 폴더에서는 저장할 때마다 최신 ONNX로 갱신한다.
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      # 메타데이터에 기록할 이름: W&B 실행 이름 또는 로컬 학습 표시.
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)  # 관측/행동·관절·제어 정보 등을 추출한다.
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      # TensorBoard 모드에서는 이 업로드 분기를 실행하지 않는다.
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
