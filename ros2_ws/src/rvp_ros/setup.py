from glob import glob

from setuptools import setup

package_name = "rvp_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saeedul Haque",
    maintainer_email="haquesaeedul@gmail.com",
    description="ROS 2 adapter for the robust_visual_perception core package.",
    license="AGPL-3.0-only",
    entry_points={
        "console_scripts": [
            "perception_node = rvp_ros.perception_node:main",
            "scripted_drive = rvp_ros.scripted_drive:main",
            "cmd_guard = rvp_ros.cmd_guard:main",
            "run_monitor = rvp_ros.run_monitor:main",
            "export_bag = rvp_ros.export_bag:main",
        ],
    },
)
