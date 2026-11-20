FROM ros:humble-ros-base-jammy

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 participant \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash participant

ENV HOME=/home/participant \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY --chmod=0755 entrypoint.sh /entrypoint.sh

USER 1000:1000
WORKDIR /home/participant
ENTRYPOINT ["/entrypoint.sh"]
