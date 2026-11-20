#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import time


EXPECTED_PING = {
    "protocol_version": "1.0",
    "supported_tasks": [],
    "self_trained_rl": False,
    "preferred_speed_mps": None,
}


def verify_ping(command):
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    elapsed = time.monotonic() - started
    result.check_returncode()
    if elapsed >= 10:
        raise ValueError(f"ping took {elapsed:.3f} s")
    payload = json.loads(result.stdout)
    if payload != EXPECTED_PING or payload["self_trained_rl"] is not False:
        raise ValueError(f"Unexpected ping payload: {payload!r}")
    print(f"ping passed in {elapsed:.3f} s")


def verify_image(image):
    platform = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", image],
        text=True,
    ).strip()
    if platform != "linux/amd64":
        raise ValueError(f"Unexpected image platform: {platform}")

    run = ["docker", "run", "--rm", "--pull=never", "--network=none"]
    verify_ping([*run, image, "ping"])
    verify_ping([*run, "--entrypoint", "/entrypoint.sh", image, "ping"])

    # Initialize a real ROS node to verify the selected RMW loads offline.
    runtime_check = """
test "$(id -u)" -ne 0
test -x /entrypoint.sh
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_ID" = 22.04
. /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
test "$RMW_IMPLEMENTATION" = rmw_cyclonedds_cpp
test "$(ros2 pkg prefix rmw_cyclonedds_cpp)" = /opt/ros/humble
python3 - <<'PYTHON'
import rclpy

rclpy.init()
node = rclpy.create_node('go1_t1_smoke_test')
node.destroy_node()
rclpy.shutdown()
PYTHON
"""
    subprocess.run(
        [*run, "--entrypoint", "/bin/bash", image, "-ec", runtime_check],
        check=True,
        timeout=30,
    )
    print("linux/amd64, Jammy, non-root, ROS 2 Humble and offline Cyclone DDS passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Locally available Docker image")
    args = parser.parse_args()
    if args.image:
        verify_image(args.image)
        return

    entrypoint = str(Path(__file__).resolve().with_name("entrypoint.sh"))
    verify_ping([entrypoint, "ping"])
    for arguments in ([], ["ping", "extra"], ["T2"], ["T3"], ["T4"], ["T5"], ["unknown"]):
        result = subprocess.run(
            [entrypoint, *arguments], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 64 or result.stdout or not result.stderr:
            raise ValueError(f"Unexpected response to {arguments!r}: {result!r}")
    print("Entrypoint error cases passed")


if __name__ == "__main__":
    main()
