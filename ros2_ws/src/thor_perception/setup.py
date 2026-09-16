from glob import glob
import os

from setuptools import find_packages, setup


package_name = "thor_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tests"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.txt") + glob("config/*.yaml") + glob("config/*.yml"),
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*.sh")),
    ],
    install_requires=["setuptools", "numpy", "thor_behavior", "thor_telemetry"],
    zip_safe=True,
    maintainer="rick98033",
    maintainer_email="rick98033@users.noreply.github.com",
    description="Archived ROS 2 face identity and person association pipeline",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_ingest = thor_perception.nodes.camera_ingest_node:main",
            "person_detector = thor_perception.nodes.person_detector_node:main",
            "person_tracker = thor_perception.nodes.person_tracker_node:main",
            "perception_state_server = thor_perception.nodes.perception_state_server:main",
            "face_detection = thor_perception.nodes.face_detection_node:main",
            "face_state_server = thor_perception.nodes.face_state_server:main",
            "auraface_id = thor_perception.nodes.auraface_id_node:main",
            "authorization = thor_perception.nodes.authorization_node:main",
        ],
    },
)
