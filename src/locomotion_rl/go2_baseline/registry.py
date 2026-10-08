"""Register the assembled Go2 flat and rough tasks with mjlab."""
# 외부 라이브러리인 mjlab의 tasks/registry.py에 정의된 함수
# /home/kante/go2_rl/setup.py 에서 mjlab==1.2.0으로 의존성을 지정함
from mjlab.tasks.registry import register_mjlab_task
from .step_03_training.runner import VelocityOnPolicyRunner
# /home/kante/go2_rl/src/locomotion_rl/go2_baseline/step_01_environment/go2_env.py
from .step_01_environment.go2_env import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .step_03_training.ppo import unitree_go2_ppo_runner_cfg

# 내부에서는 중복 등록을 검사한 뒤, 모듈의 _REGISTRY 딕셔너리(공동딕셔너리) 에 설정을 저장
# ->
# 등록이 끝나면 학습 코드(train.py) 에서 같은 라이브러리의 조회 함수를 가져와 사용합니다.
# 이 파일을 import하면 태스크 구성을 등록한다. 여기서 시뮬레이션이나 학습 루프를 실행하지는 않는다.
register_mjlab_task(
  task_id="Unitree-Go2-Rough",  # train.py/play.py 첫 인자에 입력할 험지 태스크 이름.
  env_cfg=unitree_go2_rough_env_cfg(),  # 학습용 환경 설정: 관측 노이즈·외란·커리큘럼 포함.
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),  # 평가용 환경 설정을 별도로 생성한다.
  rl_cfg=unitree_go2_ppo_runner_cfg(),  # Actor/Critic 구조, PPO 하이퍼파라미터, 로그 설정.
  runner_cls=VelocityOnPolicyRunner,  # 체크포인트 저장 시 ONNX도 내보내는 학습 실행기.
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",  # 평지 태스크. 위와 별도의 환경 설정을 등록하고 PPO 구성은 공유한다.
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# 동록된 환경설정/ PPO설정/ Runner 을 가져오고
# CLI옵션으로 이번 실행 값을 덮어 쓴다.

'''

registry.py
= "Unitree-Go2-Flat이라는 이름은
   이 환경 + 이 PPO 설정 + 이 Runner를 써라"라고 등록

train.py
= 사용자가 Unitree-Go2-Flat을 선택하면
   registry에서 그 세 가지를 찾아 실제 학습을 실행

[ 예시 ]
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard


환경 설정  → unitree_go2_flat_env_cfg()
학습 설정  → unitree_go2_ppo_runner_cfg()
학습기     → VelocityOnPolicyRunner

'''
