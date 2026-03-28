from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'AAAmodel'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # 1. 基础资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # 2. package.xml 文件
        ('share/' + package_name, ['package.xml']),
        # 3. 安装 launch 文件夹下的所有启动文件
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # 4. 安装 model 文件夹下的所有权重文件 (.pt)
        # 这一步非常关键，否则 get_package_share_directory 会找不到模型
        (os.path.join('share', package_name, 'model'), 
            glob(os.path.join('model', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yxm',
    maintainer_email='your_email@todo.todo',
    description='ROS 2 YOLO Vision and Tactical AI Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 格式：可执行文件名 = 包名.文件名:入口函数名
            'yolo_detector = AAAmodel.vision_node:main',
            'tactical_ai = AAAmodel.ai_node:main',
        ],
    },
)