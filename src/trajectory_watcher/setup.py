from setuptools import find_packages, setup

package_name = 'trajectory_watcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # This installs the package.xml and resource files.
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the launch files to share/trajectory_watcher/launch
        ('share/' + package_name + '/launch', ['launch/trajectory_watcher_launch.py']),
    ],
    install_requires=['setuptools', 'shapely'],
    zip_safe=True,
    maintainer='ganesh',
    maintainer_email='ganesh@sphaira.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_watcher_node = trajectory_watcher.trajectory_watcher_node:main',
        ],
    },
)
