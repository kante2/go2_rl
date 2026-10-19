"""시뮬레이션과 실로봇이 함께 읽는 모델별 실행 설정."""

import json
from pathlib import Path


MODEL_FOLDERS = {
    "baseline": "baseline",
    "quietpaw": "quietpaw",
    "quietpaw_long": "quiet_paw_long",
}


def is_model_config(config):
    return "model" in config and "model_dir" in config


def read_model_config(config_dir, model=None, path=None):
    """명시한 config가 없으면 모델 이름으로 JSON을 선택한다.

    모델별 JSON은 다른 모델과 조합하면 오류로 처리한다.
    """
    selected = model or "quietpaw_long"
    if selected not in MODEL_FOLDERS:
        raise ValueError(f"Unknown model: {selected}")
    path = Path(path) if path is not None else Path(config_dir) / f"{selected}.json"
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if is_model_config(config):
        if config.get("model") not in MODEL_FOLDERS:
            raise ValueError("모델별 JSON의 model이 올바르지 않습니다")
        if model is not None and config["model"] != model:
            raise ValueError(f"모델과 JSON이 다릅니다: {model} / {config['model']} ({path})")
        if "simulation" in config or "real" in config:
            raise ValueError("real/simulation 구분 없이 모든 설정을 최상위에 지정하세요")
        for field in ("model_dir", "checkpoint", "quietness"):
            if field not in config:
                raise ValueError(f"모델별 JSON에 {field}이 없습니다: {path}")
    return path, config
