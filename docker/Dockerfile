# Python and the base filesystem are pinned independently of the host OS.
FROM python:3.11.16-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b AS ml_dependencies

LABEL org.opencontainers.image.title="Go2 RL training and visualization"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONPATH=/workspace/unitree_rl_mjlab/src/go2_autonomous_driving:/workspace/unitree_rl_mjlab

# Torch/Warp wheels supply their CUDA libraries; the host supplies the driver.
# GLVND/EGL support headless MuJoCo; no X11 socket or privileged mode is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git build-essential \
    libegl1 libegl-mesa0 libgl1 libglx-mesa0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.lock /opt/go2/requirements.lock
# Cache downloads across interrupted builds without storing them in the image.
RUN --mount=type=cache,id=go2-rl-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR python -m pip install -r /opt/go2/requirements.lock \
    && python -m pip check

# ROS binaries use Ubuntu's Python 3.12. The ML runtime stays on Python 3.11.
FROM ros:jazzy-ros-base-noble@sha256:066420e07f60aa18262f2479981def87ebcfcec42eefb0c0c57c4a46098348ca
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ROS_DOMAIN_ID=42
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libegl-mesa0 libgl1 libglx-mesa0 libglib2.0-0t64 libgomp1 \
    libssl3t64 libffi8 libbz2-1.0 liblzma5 libsqlite3-0 libreadline8t64 \
    libexpat1 libuuid1 libncursesw6 libtinfo6 libcrypt1 zlib1g \
    python3-numpy python3-yaml python3-jsonschema ros-jazzy-pcl-conversions ros-jazzy-sensor-msgs-py \
    ros-jazzy-tf2-ros ros-jazzy-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Pin the same planner revision used by this project; copy only its two packages.
ARG GO2_PLANNER_REVISION=9d95a6555bbfbd27b24e9f989c0e67a8268f1500
RUN git init /opt/go2_planner_source \
    && git -C /opt/go2_planner_source remote add origin https://github.com/Yuxin916/Navigation-Physical-Experiment.git \
    && git -C /opt/go2_planner_source sparse-checkout set src/base_autonomy/local_planner src/utilities/serial \
    && git -C /opt/go2_planner_source fetch --depth 1 --filter=blob:none origin "$GO2_PLANNER_REVISION" \
    && git -C /opt/go2_planner_source checkout --detach FETCH_HEAD

COPY --from=ml_dependencies /usr/local /usr/local
COPY --from=ml_dependencies /opt/go2/requirements.lock /opt/go2/requirements.lock
COPY docker/container_env.sh /usr/local/bin/go2-env
COPY docker/ml_python.sh /usr/local/bin/go2-python
RUN chmod +x /usr/local/bin/go2-env /usr/local/bin/go2-python \
    && ln -sf go2-python /usr/local/bin/python \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3 \
    && touch /opt/go2/container

WORKDIR /workspace/unitree_rl_mjlab
COPY setup.py ./
COPY src ./src
COPY scripts ./scripts
RUN go2-python -m pip install --no-deps --no-build-isolation -e . \
    && go2-python -m pip check
# This install tree is inside the image, separate from the host's ROS build.
RUN /bin/bash -c 'source /opt/ros/jazzy/setup.bash && \
    /usr/bin/python3 scripts/build_go2_navigation.py \
    --planner-root /opt/go2_planner_source --workspace-root /opt/go2_ros_ws --system-pcl'

# Real robot ROS + ONNX share Ubuntu's Python 3.12 inside the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pytest ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/bin/python3 -m venv --system-site-packages /opt/go2_real \
    && /opt/go2_real/bin/python -m pip install 'onnxruntime==1.23.2' \
    && /opt/go2_real/bin/python -m pip check \
    && printf '#!/bin/sh\nexec /opt/go2_real/bin/python "$@"\n' > /usr/local/bin/go2-real-python \
    && chmod +x /usr/local/bin/go2-real-python
ARG GO2_UNITREE_REVISION=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
RUN git init /opt/go2_unitree_source \
    && git -C /opt/go2_unitree_source remote add origin https://github.com/unitreerobotics/unitree_ros2.git \
    && git -C /opt/go2_unitree_source sparse-checkout set cyclonedds_ws/src/unitree/unitree_go cyclonedds_ws/src/unitree/unitree_api \
    && git -C /opt/go2_unitree_source fetch --depth 1 --filter=blob:none origin "$GO2_UNITREE_REVISION" \
    && git -C /opt/go2_unitree_source checkout --detach FETCH_HEAD
RUN /bin/bash -c 'source /opt/ros/jazzy/setup.bash && \
    colcon --log-base /opt/go2_unitree_ws/log build \
    --base-paths /opt/go2_unitree_source/cyclonedds_ws/src/unitree \
    --packages-select unitree_go unitree_api --executor sequential \
    --build-base /opt/go2_unitree_ws/build --install-base /opt/go2_unitree_ws/install \
    --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3'

ENTRYPOINT ["/usr/local/bin/go2-env"]
CMD ["bash"]
