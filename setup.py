#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob 
import os
from setuptools import find_packages, setup

package_name = 'rby1_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choiyj',
    maintainer_email='cyj21c6352@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'meta_node = rby1_ros.meta_node:main',
            'rby1_control = rby1_ros.rby1_impedance_control:main',
            'main_node = rby1_ros.main_node:main',
            'tick_publisher = rby1_ros.data_tick:main',
            'realsense_record_node = rby1_ros.realsense_data_node:main',
            'zed_record_node = rby1_ros.zed_data_node:main',
            'digit_record_node = rby1_ros.digit_data_node:main',
            'rby1_record_node = rby1_ros.rby1_data_node:main',
            'zed_img_sender = rby1_ros.zed_img_sender:main',
        ],
    },
)
