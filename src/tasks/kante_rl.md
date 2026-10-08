# train


# 평지 보행 학습 명령어

<!-- 
옵션	                             의미
Unitree-Go2-Flat	                평지에서 목표 속도를 따라가는 보행 학습
--env.scene.num-envs 4096	        시뮬레이션 환경 4,096개를 병렬 실행
--agent.max-iterations 10001	    경험 수집과 PPO 업데이트를 10,001회 반복
--agent.logger tensorboard	        보상·손실 등 학습 기록 저장
--agent.run-name go2_flat	        저장 폴더에 붙일 실행 이름

저장 : /home/kante/go2_rl/logs/rsl_rl/go2_velocity/<날짜_시간>_go2_flat/ 
-->


cd /home/kante/go2_rl

python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10001 \
  --agent.logger tensorboard \
  --agent.run-name go2_flat

