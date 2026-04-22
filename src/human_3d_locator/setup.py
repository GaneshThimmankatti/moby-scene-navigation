from setuptools import setup
import os
from glob import glob

package_name = 'human_3d_locator'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'pyrealsense2', 'opencv-python'],
    zip_safe=True,
    maintainer='Ganesh Thimmankatti',
    maintainer_email='ganesh@sphaira.com',
    description='2D-to-3D detection lifting via depth image ROI projection (RealSense)',
    license='Sphaira Inc.',
    entry_points={
        'console_scripts': [
            'human_3d_locator = human_3d_locator.tracker_node_3d:main',
        ],
    },
)
