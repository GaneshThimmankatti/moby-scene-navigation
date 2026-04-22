from setuptools import setup
import os
from glob import glob

package_name = 'deep_sort_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include configuration files (e.g., params.yaml)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
    ],
    install_requires=[
        'setuptools', 'numpy', 'torch'
    ],
    zip_safe=True,
    maintainer='Ganesh Thimmankatti',
    maintainer_email='ganesh@sphaira.com',
    description='A ROS2 package for tracking using DeepSORT algorithm',
    license='Sphaira Inc.',
    entry_points={
        'console_scripts': [
            'deep_sort_tracker = deep_sort_tracker.deep_sort_tracker:main',
        ],
    },
)
