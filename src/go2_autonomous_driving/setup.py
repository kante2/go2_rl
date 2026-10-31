from glob import glob
from setuptools import find_packages, setup

package_name = "go2_autonomous_driving"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/model_pt", glob("model_pt/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "keyboard_waypoint = go2_autonomous_driving.sim.keyboard_node:main",
        "navigation_bridge = go2_autonomous_driving.sim.ros_bridge:main",
        "go2_simulation = go2_autonomous_driving.sim.simulation:main",
        "go2_sim_data_collection = go2_autonomous_driving.sim.data_collection:main",
    ]},
)
