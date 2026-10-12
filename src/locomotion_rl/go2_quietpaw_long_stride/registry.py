"""Register the assembled Go2 flat and rough tasks with mjlab."""
# 외부 라이브러리인 mjlab의 tasks/registry.py에 정의된 함수
# /home/kante/go2_rl/setup.py 에서 mjlab==1.2.0으로 의존성을 지정함
from mjlab.tasks.registry import register_mjlab_task
from .step_03_training.runner import QuietPawOnPolicyRunner
# /home/kante/go2_rl/src/locomotion_rl/go2_quietpaw_long_stride/step_01_environment/go2_env.py
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
  task_id="Unitree-Go2-QuietPaw-LongStride-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),  # 학습용 환경 설정: 관측 노이즈·외란·커리큘럼 포함.
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),  # 평가용 환경 설정을 별도로 생성한다.
  rl_cfg=unitree_go2_ppo_runner_cfg(),  # Actor/Critic 구조, PPO 하이퍼파라미터, 로그 설정.
  runner_cls=QuietPawOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-QuietPaw-LongStride-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=QuietPawOnPolicyRunner,
)

