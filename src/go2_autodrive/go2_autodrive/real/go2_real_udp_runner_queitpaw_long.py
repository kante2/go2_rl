r"""QuietPaw LongStride 학습 모델로 Go2 실로봇을 제어한다 (48 입력 / 12 출력).

실행 흐름: main() -> init() -> run_robot() -> close().
기본 모델 폴더: go2_autodrive/config/model_pt/quiet_paw_long
기본 설정 파일: go2_autodrive/config/quietpaw_long.json

저장소 루트에서 모델 준비 (로봇 연결 없음)::

    python3 src/go2_autodrive/go2_autodrive/real/go2_real_udp_runner_queitpaw_long.py --prepare

실측 센서로 추론 확인 (모터 출력 없음)::

    bash scripts/run_go2_real_docker.sh \
      src/go2_autodrive/go2_autodrive/real/go2_real_udp_runner_queitpaw_long.py --seconds 10

실제 제어는 --execute를 추가하고 ARM을 입력한다.
PT만 가져오면 ONNX를 생성한다. policy.onnx도 있으면 PT와 일치하는지 검사한다.
여러 PT가 있으면 model_<번호>.pt 중 가장 큰 번호를 선택한다.
특정 모델은 --checkpoint로 지정한다 (CLI 경로는 현재 작업 폴더 기준).
--export-onnx는 가져온 ONNX 대신 PT에서 새로 변환할 때 --prepare와 함께 사용한다.
원본 파일은 수정하지 않고 검증한 파일을 artifacts/go2_real/quietpaw_long에 저장한다.
"""

import argparse
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

if __package__:
    from .go2_real_udp_runner import ROOT, REAL_CONFIG, check_real_config, sha256, read_model_config, is_model_config
else:
    from go2_real_udp_runner import ROOT, REAL_CONFIG, check_real_config, sha256, read_model_config, is_model_config


DEFAULT_CONFIG = ROOT / "src/go2_autodrive/config/quietpaw_long.json"


# ============================================================
# 1. 실행 옵션과 설정 파일
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quietness", type=float, help="0: 느슨한 소음 제약, 1: 엄격한 제약")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--export-onnx", action="store_true",
                        help="--prepare 시 가져온 ONNX 대신 PT에서 변환")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="모델 준비와 검증 후 종료")
    mode.add_argument("--check-only", action="store_true", help="준비된 모델 검사 후 종료")
    mode.add_argument("--execute", action="store_true", help="ARM 입력 후 모터 제어")
    args = parser.parse_args(argv)
    if args.export_onnx and not args.prepare:
        parser.error("--export-onnx는 --prepare와 함께 사용하세요")
    if args.seconds is not None:
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            parser.error("--seconds must be positive and finite")
    return args


def select_checkpoint(model_dir):
    """파일 수정 시간이 아닌 학습 반복 번호로 최신 PT를 선택한다."""
    candidates = []
    for path in model_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"model_<번호>.pt가 없습니다: {model_dir}")
    return max(candidates)[1]


def load_config(args):
    config_path, config = read_model_config(DEFAULT_CONFIG.parent, "quietpaw_long", args.config)
    model_dir = (config_path.parent / config["model_dir"]).resolve()

    # CLI는 현재 작업 폴더, JSON의 checkpoint는 model_dir 기준이다.
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
    elif config.get("checkpoint"):
        checkpoint = (model_dir / config["checkpoint"]).resolve()
    else:
        checkpoint = select_checkpoint(model_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    quietness = args.quietness if args.quietness is not None else config["quietness"]
    if type(quietness) not in (int, float) or not math.isfinite(quietness) or not 0 <= quietness <= 1:
        raise ValueError("quietness must be finite and in [0, 1]")

    real_config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    if is_model_config(config):
        overrides = {key: config[key] for key in real_config if key in config}
    else:
        overrides = config.get("real", {})
    if not isinstance(overrides, dict) or overrides.keys() - real_config.keys():
        raise ValueError('"real"에는 기존 실로봇 제어 설정만 지정하세요')
    real_config.update(overrides)
    check_real_config(real_config)

    # ONNX를 함께 가져왔으면 검증 대상으로 사용한다. PT만 있어도 준비 가능하다.
    source_onnx = checkpoint.parent / "policy.onnx"
    if args.export_onnx or not source_onnx.is_file():
        source_onnx = None
    return real_config, checkpoint, source_onnx, float(quietness)


# ============================================================
# 2. 초기화: 모델 선택, 준비된 파일 검사, 임시 제어 설정
# ============================================================
def init(args):
    config, checkpoint, source_onnx, quietness = load_config(args)
    policy_dir = ROOT / "artifacts/go2_real/quietpaw_long" / sha256(checkpoint)
    state = {
        "config": config,
        "checkpoint": checkpoint,
        "source_onnx": source_onnx,
        "policy_dir": policy_dir,
        "quietness": quietness,
        "temp_directory": None,
        "config_path": None,
    }
    try:
        print(f"Checkpoint: {checkpoint}\nPolicy: {policy_dir}", flush=True)
        if args.prepare:
            prepare_policy(state)
        metadata = check_policy(state)
        print(
            f"QuietPaw LongStride: 48 -> 12, quietness={quietness:g}, "
            f"step_dt={metadata['step_dt']:g}, phase_period={metadata['phase_period']:g}",
            flush=True,
        )
        if args.prepare or args.check_only:
            return state

        directory = tempfile.TemporaryDirectory(prefix="go2-quietpaw-long-")
        state["temp_directory"] = directory
        config_path = Path(directory.name) / "real.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        state["config_path"] = config_path
        return state
    except BaseException:
        close(state)
        raise


# ============================================================
# 3. 모델 준비: 학습 설정 복원 -> ONNX 준비 -> 검증 후 저장
# ============================================================
def prepare_policy(state):
    checkpoint = state["checkpoint"]
    policy_dir = state["policy_dir"]
    try:
        checkpoint_arg = str(checkpoint.relative_to(ROOT))
    except ValueError as error:
        raise ValueError("모델은 Docker에 마운트된 저장소 안에 두세요") from error

    # 검증 실패 시 이전에 준비한 모델을 건드리지 않는다.
    # artifacts는 Docker(root) 소유일 수 있으므로 임시 폴더는 저장소 루트에 만든다.
    with tempfile.TemporaryDirectory(prefix=".go2-long-prepare-", dir=ROOT) as directory:
        stage = Path(directory)
        stage_arg = str(stage.relative_to(ROOT))
        wrapper = ["bash", str(ROOT / "scripts/run_go2_docker.sh")]
        command = wrapper + [
            "python", "scripts/export_go2_quietpaw_long_policy.py",
            "--checkpoint", checkpoint_arg,
            "--output", stage_arg,
        ]
        if state["source_onnx"] is not None:
            command += ["--source-onnx", str(state["source_onnx"].relative_to(ROOT))]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(
            wrapper + ["go2-real-python", "scripts/check_go2_real_policy.py",
                       "--policy-dir", stage_arg],
            cwd=ROOT, check=True,
        )

        # 메타데이터는 마지막에 저장한다. 부분 교체는 hash 검사에서 거부된다.
        publish = """
from pathlib import Path
import sys
stage, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
for name in ('policy.onnx', 'parity.npz', 'samples.json', 'policy.json'):
    (stage / name).replace(output / name)
"""
        subprocess.run(
            wrapper + ["python", "-c", publish, stage_arg, str(policy_dir.relative_to(ROOT))],
            cwd=ROOT, check=True,
        )


# ============================================================
# 4. 실행 전 검사: 모델 변경, 정책 종류, LongStride 주기 확인
# ============================================================
def check_policy(state):
    directory = state["policy_dir"]
    metadata_path = directory / "policy.json"
    policy_path = directory / "policy.onnx"
    if not metadata_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("준비된 모델이 없습니다. 먼저 --prepare를 실행하세요")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["checkpoint_sha256"] != sha256(state["checkpoint"]):
        raise ValueError("PT가 변경되었습니다. --prepare를 다시 실행하세요")
    if metadata["onnx_sha256"] != sha256(policy_path):
        raise ValueError("ONNX가 변경되었습니다. --prepare를 다시 실행하세요")
    source = state["source_onnx"]
    if source is not None and metadata.get("source_onnx_sha256") is not None:
        if metadata["source_onnx_sha256"] != sha256(source):
            raise ValueError("가져온 ONNX가 변경되었습니다. --prepare를 다시 실행하세요")
    if metadata.get("policy_variant") != "quietpaw_long_stride":
        raise ValueError("QuietPaw LongStride 메타데이터가 아닙니다")
    for name in ("step_dt", "phase_period"):
        if not math.isfinite(metadata[name]) or metadata[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if metadata["phase_period"] != metadata["stride"]["gait_period"]:
        raise ValueError("학습 보행 주기와 실로봇 위상 주기가 다릅니다")

    # 이 매핑은 ROS 없이도 48개 관측값 구성과 소음 조건 범위를 검사한다.
    sys.path.insert(0, str(ROOT / "src/go2_autonomous_driving"))
    from go2_autonomous_driving.real.real_quietpaw import QuietPawMapping

    QuietPawMapping(metadata, state["quietness"])
    return metadata


# ============================================================
# 5. 실로봇 실행: 실측 47개 관측값 + 소음 조건 1개 -> 관절 12개
# ============================================================
def run_robot(args, state):
    from go2_autonomous_driving.real.real_closed_loop import ClosedLoopBackend
    from go2_autonomous_driving.real.real_quietpaw import QuietPawMapping
    from go2_autonomous_driving.real.real_robot import run

    controller_args = argparse.Namespace(
        checkpoint=state["checkpoint"],
        policy_dir=state["policy_dir"],
        config=state["config_path"],
        execute=args.execute,
        seconds=args.seconds,
        fault_log=ROOT / "log/go2_quietpaw_long_fault.json",
    )
    # 기존 ARM, 센서 검사, 키보드 timeout, 비상정지, 종료 damping을 재사용한다.
    run(
        controller_args,
        backend_type=ClosedLoopBackend,
        requires_navigation=False,
        send_telemetry=True,
        mapping_factory=lambda metadata: QuietPawMapping(metadata, state["quietness"]),
    )


# ============================================================
# 6. 종료 및 시작점
# ============================================================
def close(state):
    if state is not None and state["temp_directory"] is not None:
        state["temp_directory"].cleanup()
        state["temp_directory"] = None


def interrupted(*unused):
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    state = None
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        state = init(args)
        if args.prepare or args.check_only:
            print("검사 완료. 로봇 연결 및 모터 출력 없음.", flush=True)
            return
        run_robot(args, state)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    finally:
        close(state)


if __name__ == "__main__":
    main()
