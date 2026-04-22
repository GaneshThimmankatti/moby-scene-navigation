from setuptools import setup

package_name = 'mot17_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ganesh Thimmankatti',
    maintainer_email='ganesh@sphaira.com',
    description='Publisher node for MOT17 dataset',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'mot17_image_publisher = mot17_publisher.mot17_image_publisher:main',
            'gt_publisher = mot17_publisher.gt_publish:main',
        ],
    },
)
