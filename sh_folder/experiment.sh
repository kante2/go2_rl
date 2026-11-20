#!/usr/bin/env bash
# 실험 설정: 각 터미널에서 먼저 `source /home/kante/go2_rl/sh_folder/experiment.sh` 한 번.
# 조건을 바꾸려면 아래 값을 고치고 모든 터미널에서 다시 source 한다.
# 터미널에서 잠깐 바꾸려면 source 전에 지정해도 된다: SPEED=0.7 source .../experiment.sh

GO2_ROOT=/home/kante/go2_rl

MODEL=${MODEL:-34997}                 # 보행 모델 번호 (src/go2_autonomous_driving/model_pt/model_<번호>.pt)
SPEED=${SPEED:-0.5}                   # 전진·후진 속도 [m/s]      (W/X, ↑/↓)
LATERAL_SPEED=${LATERAL_SPEED:-0.5}   # 횡방향 속도 [m/s]        (A/D, ←/→)
YAW_RATE=${YAW_RATE:-0.3}             # 제자리 회전 속도 [rad/s]  (Q/E)
DURATION=${DURATION:-30}              # 녹화 시간 [초] (S3·R4 --duration, 지나면 자동 종료·저장)
LABEL=${LABEL:-NORMAL}                # 파일 이름 앞부분
DATA_DIR=${DATA_DIR:-src/sim_to_real_data/real_sim}   # 녹화 저장 폴더 (프로젝트 기준)

# 녹화 이름: <LABEL>_<SPEED>_<LATERAL_SPEED>_<YAW_RATE>_v<다음 번호>
# DATA_DIR에 저장된 같은 조건의 파일을 보고 번호를 매긴다. S3·R4를 연달아 켜면 같은 번호가 된다.
exp_base() { echo "${LABEL}_${SPEED}_${LATERAL_SPEED}_${YAW_RATE}"; }   # record_session.sh --base
exp_name() {
  local base n=0 f v
  base="$(exp_base)"
  for f in "$GO2_ROOT/$DATA_DIR/${base}"_v*; do
    [[ -e "$f" ]] || continue
    v="${f##*/"${base}"_v}"
    v="${v%%_*}"
    [[ "$v" =~ ^[0-9]+$ ]] && (( 10#$v > n )) && n=$((10#$v))
  done
  echo "${base}_v$((n + 1))"
}

echo "실험 설정: MODEL=$MODEL SPEED=$SPEED LATERAL_SPEED=$LATERAL_SPEED YAW_RATE=$YAW_RATE DURATION=${DURATION}s → 다음 녹화 이름 $(exp_name)"
