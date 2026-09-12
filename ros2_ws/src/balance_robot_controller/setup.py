from setuptools import setup
from glob import glob
import os

package_name = 'balance_robot_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # Ament index marker
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # Package.xml
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='PID balance controller with auto-tuning for self-balancing robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'balance_controller = balance_robot_controller.balance_controller:main',
            'pid_tuner = balance_robot_controller.pid_tuner:main',
        ],
    },
)
