r"""모델 종류를 선택해 Go2 실로봇을 제어한다.

실행 흐름: main() -> init() -> run_robot() -> close().
--model: baseline / quietpaw / quietpaw_long. 기본 설정은 config/<모델>.json.
--prepare: ONNX/제어 정보 준비 및 검증. --check-only: 준비된 모델 검사.
기본 실행은 센서/추론만 확인한다. --execute와 ARM 입력 후 모터가 활성화된다.
--keyboard는 동일 config의 포트와 속도로 실로봇 키보드를 실행한다.
PT는 config/model_pt/<모델 폴더>에서 가장 큰 학습 번호를 선택한다.
--checkpoint는 현재 작업 폴더 기준 경로, JSON의 checkpoint는 모델 폴더 기준이다.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "src/go2_autodrive/config"
# 파일 직접 실행과 ROS 패키지 실행에서 같은 설정 로더를 사용한다.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from go2_autodrive.model_config import read_model_config, is_model_config
else:
    from ..model_config import read_model_config, is_model_config
REAL_CONFIG = ROOT / "src/go2_autonomous_driving/config/real_closed_loop.json"
MODELS = {
    "baseline": ("baseline", "baseline", 47),
    "quietpaw": ("quietpaw", "quietpaw", 48),
    "quietpaw_long": ("quiet_paw_long", "quietpaw_long_stride", 48),
}
SPEED_OPTIONS = (
    ("speed", "keyboard_forward_speed", "max_forward_speed"),
    ("lateral_speed", "keyboard_lateral_speed", "max_lateral_speed"),
    ("yaw_rate", "keyboard_yaw_rate", "max_yaw_rate"),
)


# ============================================================
# 1. 실행 옵션과 설정 파일
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quietness", type=float, help="0: 느슨한 소음 제약, 1: 엄격한 제약")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--speed", type=float, help="W/S 전후 목표 속도 [m/s]")
    parser.add_argument("--lateral-speed", type=float, help="A/D 좌우 목표 속도 [m/s]")
    parser.add_argument("--yaw-rate", type=float, help="J/L 회전 목표 속도 [rad/s]")
    parser.add_argument("--export-onnx", action="store_true",
                        help="--prepare 시 가져온 ONNX 대신 PT에서 변환")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="모델 준비와 검증 후 종료")
    mode.add_argument("--check-only", action="store_true", help="준비된 모델 검사 후 종료")
    mode.add_argument("--execute", action="store_true", help="ARM 입력 후 모터 제어")
    mode.add_argument("--keyboard", action="store_true", help="동일 config로 실로봇 키보드 실행")
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
    config_path, config = read_model_config(CONFIG_DIR, args.model, args.config)
    print(f"Config: {config_path}", flush=True)
    model = args.model or config["model"]
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if is_model_config(config):
        model_dir = (config_path.parent / config["model_dir"]).resolve()
    else:
        model_dir = (config_path.parent / config["model_root"] / MODELS[model][0]).resolve()
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
    for option, key, limit in SPEED_OPTIONS:
        value = getattr(args, option)
        if value is not None:
            if not math.isfinite(value) or not 0 < value <= real_config[limit]:
                raise ValueError(f"{option}은 0보다 크고 {limit}={real_config[limit]:g} 이하여야 합니다")
            real_config[key] = value

    checkpoint = None
    source_onnx = None
    if not args.keyboard:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint.resolve()
        elif config.get("checkpoint") and model == config["model"]:
            checkpoint = (model_dir / config["checkpoint"]).resolve()
        else:
            checkpoint = select_checkpoint(model_dir)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        candidate = checkpoint.parent / "policy.onnx"
        if candidate.is_file() and not args.export_onnx:
            source_onnx = candidate
    return real_config, model, checkpoint, source_onnx, float(quietness)


def check_real_config(config):
    """포트, 속도 제한, 타임아웃을 로봇 연결 전에 검사한다."""
    for key, value in config.items():
        if key.endswith("_port"):
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f"{key} must be an integer in [1, 65535]")
            continue

        # 가속도/감속도는 [전후, 좌우, 회전]의 세 값이다.
        if key in ("command_acceleration", "command_deceleration"):
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{key} must contain three values")
            values = value
        else:
            values = [value]

        for number in values:
            if type(number) not in (int, float):
                raise ValueError(f"{key} must be positive and finite")
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{key} must be positive and finite")

    ports = [
        config["command_port"],
        config["telemetry_port"],
        config["keyboard_status_port"],
    ]
    if len(set(ports)) != len(ports):
        raise ValueError("실로봇 UDP 포트는 서로 달라야 합니다")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# 2. 초기화: 모델 선택, 준비된 파일 검사, 임시 제어 설정
# ============================================================
def init(args):
    config, model, checkpoint, source_onnx, quietness = load_config(args)
    policy_dir = None
    if checkpoint is not None:
        policy_dir = ROOT / "artifacts/go2_real" / model / sha256(checkpoint)
    state = {
        "config": config,
        "model": model,
        "checkpoint": checkpoint,
        "source_onnx": source_onnx,
        "policy_dir": policy_dir,
        "quietness": quietness,
        "temp_directory": None,
        "config_path": None,
        "keyboard_speed_overrides": {
            option: getattr(args, option) for option, _, _ in SPEED_OPTIONS
            if getattr(args, option) is not None
        },
    }
    try:
        print(
            f"Keyboard defaults: W/S={config['keyboard_forward_speed']:g} m/s, "
            f"A/D={config['keyboard_lateral_speed']:g} m/s, "
            f"J/L={config['keyboard_yaw_rate']:g} rad/s", flush=True,
        )
        if not args.keyboard:
            print(f"Model: {model}\nCheckpoint: {checkpoint}\nPolicy: {policy_dir}", flush=True)
            if args.prepare:
                prepare_policy(state)
            metadata = check_policy(state)
            mapping = make_mapping(state, metadata)
            condition = ""
            if model != "baseline":
                condition = f", quietness={quietness:g}, cost_limit={mapping.cost_limit:g}"
            print(
                f"{model}: {mapping.observation_size} -> 12, "
                f"step_dt={metadata['step_dt']:g}, phase_period={metadata['phase_period']:g}{condition}",
                flush=True,
            )
            if args.prepare or args.check_only:
                return state

        directory = tempfile.TemporaryDirectory(prefix="go2-real-")
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
    with tempfile.TemporaryDirectory(prefix=".go2-real-prepare-", dir=ROOT) as directory:
        stage = Path(directory)
        stage_arg = str(stage.relative_to(ROOT))
        wrapper = ["bash", str(ROOT / "scripts/run_go2_docker.sh")]
        command = wrapper + [
            "python", "scripts/prepare_go2_model_policy.py",
            "--model", state["model"],
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
# 4. 실행 전 검사: 파일 hash, 모델 종류, 입력 크기, 보행 주기
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
    expected_variant = MODELS[state["model"]][1]
    if metadata.get("policy_variant") != expected_variant:
        raise ValueError("선택한 모델 종류와 준비된 정책이 다릅니다")
    if metadata.get("observation_size") != MODELS[state["model"]][2]:
        raise ValueError("선택한 모델과 관측 입력 크기가 다릅니다")
    for name in ("step_dt", "phase_period"):
        if not math.isfinite(metadata[name]) or metadata[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if state["model"] == "quietpaw_long":
        if metadata["phase_period"] != metadata["stride"]["gait_period"]:
            raise ValueError("학습 보행 주기와 실로봇 위상 주기가 다릅니다")
    make_mapping(state, metadata)

    # 파일 검사 모드에서도 실제 ONNX 입력/출력 크기를 확인한다. ROS는 사용하지 않는다.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].shape != [1, metadata["observation_size"]]
            or inputs[0].type != "tensor(float)" or len(outputs) != 1
            or outputs[0].shape != [1, 12] or outputs[0].type != "tensor(float)"):
        raise ValueError("ONNX 입력/출력 구조가 선택한 제어기와 다릅니다")
    return metadata


def make_mapping(state, metadata):
    sys.path.insert(0, str(ROOT / "src/go2_autonomous_driving"))
    from go2_autonomous_driving.real.real_control import PolicyMapping
    from go2_autonomous_driving.real.real_quietpaw import QuietPawMapping

    if state["model"] == "baseline":
        return PolicyMapping(metadata)
    return QuietPawMapping(metadata, state["quietness"])


# ============================================================
# 5. 실로봇 실행: 모델별 관측값 구성과 키보드 연결
# ============================================================
def run_robot(args, state):
    from go2_autonomous_driving.real.real_closed_loop import ClosedLoopBackend
    from go2_autonomous_driving.real.real_robot import run

    controller_args = argparse.Namespace(
        checkpoint=state["checkpoint"],
        policy_dir=state["policy_dir"],
        config=state["config_path"],
        execute=args.execute,
        seconds=args.seconds,
        fault_log=ROOT / f"log/go2_real_{state['model']}_fault.json",
    )
    # 기존 ARM, 센서 검사, 키보드 timeout, 비상정지, 종료 damping을 재사용한다.
    run(
        controller_args,
        backend_type=ClosedLoopBackend,
        requires_navigation=False,
        send_telemetry=True,
        mapping_factory=lambda metadata: make_mapping(state, metadata),
    )


def run_keyboard(state):
    # 같은 병합 config를 전달해 컨트롤러와 키보드의 포트/속도가 일치하게 한다.
    env = dict(os.environ)
    package_path = str(ROOT / "src/go2_autonomous_driving")
    env["PYTHONPATH"] = package_path + ":" + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "go2_autonomous_driving.real.keyboard_velocity",
               "--config", str(state["config_path"])]
    # 명시한 키보드 인자만 제어기가 보내는 기본 속도보다 우선한다.
    for name, value in state.get("keyboard_speed_overrides", {}).items():
        command += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(command, env=env, check=True)


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
        if (args.execute or args.keyboard) and not sys.stdin.isatty():
            raise RuntimeError("모터 ARM/키보드 입력은 대화형 터미널에서 실행하세요")
        state = init(args)
        if args.prepare or args.check_only:
            print("검사 완료. 로봇 연결 및 모터 출력 없음.", flush=True)
            return
        if args.keyboard:
            run_keyboard(state)
        else:
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
