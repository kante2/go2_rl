# train


# 1-1.평지 보행 학습 명령어
##  옵션
옵션	                             의미
Unitree-Go2-Flat	                평지에서 목표 속도를 따라가는 보행 학습
--env.scene.num-envs 4096	        시뮬레이션 환경 4,096개를 병렬 실행
--agent.max-iterations 10001	    경험 수집과 PPO 업데이트를 10,001회 반복
--agent.logger tensorboard	        보상·손실 등 학습 기록 저장
--agent.run-name go2_flat	        저장 폴더에 붙일 실행 이름

저장 : /home/kante/go2_rl/logs/rsl_rl/go2_velocity/<날짜_시간>_go2_flat/ 



cd /home/kante/go2_rl

python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10001 \
  --agent.logger tensorboard \
  --agent.run-name go2_flat



# 1-2 체크포인트 보행 학습 명령어
cd /workspace/unitree_rl_mjlab

go2-python scripts/train.py Unitree-Go2-Flat \
  --agent.resume True \
  --agent.load-run resume_model_10000 \
  --agent.load-checkpoint model_10000.pt \
  --agent.max-iterations 5000 \
  --agent.run-name baseline_resume \
  --agent.logger tensorboard \
  --agent.save-interval 500 \
  --env.scene.num-envs 1024

# 2. 보상설계


# 3. 학습 알고리즘


# 4. Quiet Paw 의 구조 분석
하나의 보행 정책에 소음 제약값을 입력하여, 실행 중에도 조용하게 걷는 정도를 조절

발접촉력/ 착지속도 -> 소음 대리 비용

