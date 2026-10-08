from glob import glob
from setuptools import find_packages, setup

package_name = 'go2_autodrive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('go2_autodrive/launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'PyYAML', 'jsonschema>=4.0'],
    zip_safe=False,
    maintainer='kante',
    maintainer_email='kante@example.com',
    description='Go2 autonomous driving components, starting with the MuJoCo UDP to ROS bridge.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'udp_ros_bridge = go2_autodrive.udp_ros_bridge.udp_ros_bridge:main',
            'keyboard_waypoint = go2_autodrive.keyboard.keyboard_waypoint:main',
        ],
    },
)
