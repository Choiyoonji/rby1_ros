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
        ('share/' + package_name + '/msg', ['msg/EEpos.msg', 'msg/FTsensor.msg', 'msg/State.msg', 'msg/Command.msg']),
        ('share/' + package_name + '/srv', ['srv/MetaData.srv', 'srv/MetaInitial.srv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choiyj',
    maintainer_email='cyj21c6352@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'meta_node = rby1_ros.meta_node:main',
            'rby1_control = rby1_ros.rby1_impedance_control:main',
        ],
    },
)
