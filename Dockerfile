FROM ros:humble-ros-base-jammy AS interface-build

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3-colcon-common-extensions ros-humble-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*
COPY unitree_go/ /build/src/unitree_go/
WORKDIR /build
RUN /bin/bash -c 'source /opt/ros/humble/setup.bash && colcon build --merge-install --install-base /opt/go1 --packages-select unitree_go'

FROM ros:humble-ros-base-jammy

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-rmw-cyclonedds-cpp ros-humble-rosgraph-msgs python3-pip libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir numpy==1.26.4 mujoco==3.3.7 \
    && groupadd --gid 1000 participant \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash participant

ENV HOME=/home/participant \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    MUJOCO_GL=disable \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=interface-build /opt/go1/ /opt/go1/
COPY unitree_go/LICENSE /usr/local/share/doc/robprik-go1/unitree-LICENSE.txt
COPY model/ /opt/t2/model/
COPY collision.py safety.py safety_node.py test_safety.py /opt/t2/
COPY --chmod=0755 entrypoint.sh /entrypoint.sh
RUN python3 /opt/t2/model/prepare.py /opt/t2/model

USER 1000:1000
WORKDIR /home/participant
RUN /bin/bash -c 'source /opt/ros/humble/setup.bash && source /opt/go1/local_setup.bash && python3 /opt/t2/test_safety.py --native'
ENTRYPOINT ["/entrypoint.sh"]
