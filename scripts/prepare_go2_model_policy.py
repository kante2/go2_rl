"""모델 종류를 확인하고 실로봇용 파일을 준비한다. Docker ML Python 전용.

출력은 임시 폴더에만 저장한다. 호출자가 check_go2_real_policy.py 검증 후 공개한다.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from src.go2_autodrive.go2_autodrive.simulation.go2_mujuco_udp_runner import check_policy_shape
from export_go2_quietpaw_long_policy import check_source_metadata


EXPORTERS = {
    "baseline": "export_go2_real_policy.py",
    "quietpaw": "export_go2_quietpaw_policy.py",
    "quietpaw_long": "export_go2_quietpaw_long_policy.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(EXPORTERS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-onnx", type=Path)
    args = parser.parse_args()

    # 폴더 이름만 믿지 않고 PT 내부의 관측 크기와 학습 종류를 확인한다.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check_policy_shape(checkpoint, args.model)
    command = [
        sys.executable, str(Path(__file__).parent / EXPORTERS[args.model]),
        "--checkpoint", str(args.checkpoint), "--output", str(args.output),
    ]
    if args.model == "quietpaw_long" and args.source_onnx is not None:
        command += ["--source-onnx", str(args.source_onnx)]
    subprocess.run(command, check=True)

    metadata_path = args.output / "policy.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["policy_family"] = "baseline" if args.model == "baseline" else "quietpaw"
    metadata["policy_variant"] = "quietpaw_long_stride" if args.model == "quietpaw_long" else args.model
    metadata["observation_size"] = 47 if args.model == "baseline" else 48
    metadata["normalization_included"] = True
    if args.source_onnx is not None:
        check_source_metadata(args.source_onnx, metadata)
        shutil.copyfile(args.source_onnx, args.output / "policy.onnx")
        metadata["source_onnx_sha256"] = hashlib.sha256(args.source_onnx.read_bytes()).hexdigest()
    metadata["onnx_sha256"] = hashlib.sha256((args.output / "policy.onnx").read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
