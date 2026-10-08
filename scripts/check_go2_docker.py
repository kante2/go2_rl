"""Check container dependencies, Go2 model and headless rendering, without motors."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    # ROS's compiled Python modules must use Ubuntu's system interpreter.
    subprocess.run([
        "/usr/local/bin/go2-env", "/usr/bin/python3", "-c",
        "import sys, rclpy, sensor_msgs_py.point_cloud2; "
        "from pathlib import Path; "
        "from ament_index_python.packages import get_package_prefix; "
        "p=Path(get_package_prefix('local_planner')); "
        "assert (p/'lib/local_planner/localPlanner').is_file(); "
        "assert (p/'lib/local_planner/pathFollower').is_file(); "
        "get_package_prefix('go2_autonomous_driving'); "
        "print('ROS imports/planner OK: Python',sys.version.split()[0])",
    ], check=True)
    import mujoco
    import torch
    import warp as wp
    import src.locomotion_rl  # Register the training tasks.
    from mjlab.tasks.registry import list_tasks

    for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "viser"):
        print(f"{name}={importlib.metadata.version(name)}")
    assert "Unitree-Go2-Flat" in list_tasks(), "Go2 training task was not registered"
    model_path = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_go2/xmls/go2.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
        assert frame.shape == (64, 64, 3)
    print(f"Go2 model: nq={model.nq}, nu={model.nu}; EGL render OK")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}; torch CUDA version: {torch.version.cuda}")
    if args.require_cuda and not available:
        raise SystemExit("GPU check failed. Check host nvidia-smi and NVIDIA Container Toolkit.")
    if available:
        torch.ones(4, device="cuda").square().sum().item()
        wp.init()
        wp.zeros(4, device="cuda:0").numpy()
        print(f"Torch/Warp CUDA allocation OK: {torch.cuda.get_device_name(0)}")
    print("Environment check passed (no motor output).")


if __name__ == "__main__":
    main()
