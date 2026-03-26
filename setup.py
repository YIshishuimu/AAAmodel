from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_arms'

setup(
    name=package_name,
    # ... 其他配置
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 关键：把 launch 文件夹里的所有 .py 文件安装到 share 路径下
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    entry_points={
        'console_scripts': [
            'yolo_detector = my_robot_arms.vision_node:main',
            'tactical_ai = my_robot_arms.ai_node:main',
        ],
    },
)