from glob import glob
from setuptools import find_packages, setup

setup(
    name="go2_gait_monitor",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={"go2_gait_monitor": ["web/*"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/go2_gait_monitor"]),
        ("share/go2_gait_monitor", ["package.xml"]),
        ("share/go2_gait_monitor/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="kante",
    maintainer_email="kante@example.com",
    description="Go2 live gait, ground force and landing impulse dashboard",
    license="Apache-2.0",
    entry_points={"console_scripts": ["dashboard = go2_gait_monitor.server:main"]},
)
