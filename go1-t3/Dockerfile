# syntax=docker/dockerfile:1
# Participant image: Go1 RL walking policy (T3) + Safety Block.
# linux/amd64, Ubuntu 22.04, ROS 2 Humble, Cyclone DDS, CPU only, runs as non-root user "ros",
# no internet needed at run time (all dependencies, the model and the policy weights are inside).
ARG BASE_IMAGE=registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
# same unitree_ros2 revision as the official simulator (unitree_go message definitions)
ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88

USER root
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3-pip python3-numpy python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rosidl-default-generators \
        ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
        ros-${ROS_DISTRO}-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo 3.3.7 (same version as the simulator) for the self-collision checks of the Safety Block
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && python3 -c "import numpy, mujoco; assert int(numpy.__version__.split('.')[0]) < 2; assert mujoco.__version__ == '3.3.7'"

# unitree_go messages
RUN mkdir -p /opt/unitree_ws/src && cd /tmp \
    && git clone https://github.com/unitreerobotics/unitree_ros2.git \
    && cd unitree_ros2 && git checkout ${UNITREE_ROS2_REF} \
    && cp -r cyclonedds_ws/src/unitree/unitree_go /opt/unitree_ws/src/ \
    && cd /opt/unitree_ws && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --packages-select unitree_go --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log /tmp/unitree_ros2 \
    && source /opt/unitree_ws/install/setup.bash \
    && python3 -c "from unitree_go.msg import LowCmd, LowState; assert len(LowCmd().motor_cmd) == 20"

# solution
COPY go1_solution /opt/go1_solution/go1_solution
COPY models /opt/go1_solution/models
COPY policy /opt/go1_solution/policy
COPY ping.json /opt/go1_solution/ping.json
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh && chmod -R a+rX /opt/go1_solution /opt/unitree_ws \
    && sha256sum /opt/go1_solution/models/unitree_go1/go1.xml | grep -q d5a7466784c8e72fd174cc7fa5a92754a1a0a6d458a1dec79e7ca0abc9525c4a \
    && test -f /opt/go1_solution/policy/policy.npz

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    PYTHONPATH=/opt/go1_solution \
    PYTHONUNBUFFERED=1 \
    GO1_SOLUTION_MODEL_DIR=/opt/go1_solution/models/unitree_go1 \
    GO1_POLICY_PATH=/opt/go1_solution/policy/policy.npz \
    HOME=/tmp

USER ros
WORKDIR /tmp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ping"]
