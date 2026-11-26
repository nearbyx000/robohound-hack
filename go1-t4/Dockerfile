# syntax=docker/dockerfile:1
FROM registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1@sha256:4d4a184901f154849d59477a077ce507f31fdf63b8bedd8caea460607eec243d

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-numpy python3-scipy ros-humble-nav-msgs ros-humble-rmw-cyclonedds-cpp \
       ros-humble-rosgraph-msgs ros-humble-sensor-msgs ros-humble-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

COPY node.py odometry.py /opt/go1-t4/
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

USER ros
WORKDIR /tmp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ping"]
