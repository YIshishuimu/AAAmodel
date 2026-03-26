import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 定义视觉识别节点 (YOLO)
    vision_node = Node(
        package='my_robot_arms',      # 功能包名
        executable='yolo_detector',    # setup.py 中定义的 entry_point 名字
        name='vision_processor',       # 启动后的节点名
        output='screen',               # 将日志打印到终端
        parameters=[{
            'conf_threshold': 0.5,     # 可以在这里传递参数，无需改源码
        }]
    )

    # 2. 定义 AI 决策节点 (Alpha-Beta)
    ai_node = Node(
        package='my_robot_arms',
        executable='tactical_ai',      # setup.py 中定义的 entry_point 名字
        name='ai_engine',
        output='screen'
    )

    # 3. 如果你有现成的摄像头驱动（如 usb_cam），也可以顺便拉起来
    # camera_node = Node(
    #     package='usb_cam',
    #     executable='usb_cam_node_exe',
    #     name='camera_driver'
    # )

    # 返回启动描述，ROS 2 会按顺序启动它们
    return LaunchDescription([
        vision_node,
        ai_node
        # camera_node
    ])