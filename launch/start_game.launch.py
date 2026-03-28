import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 视觉节点
    vision_node = Node(
        package='AAAmodel',      # <--- 必须改这里！
        executable='yolo_detector',
        name='vision_processor',
        output='screen'
    )

    # 2. AI 决策节点
    ai_node = Node(
        package='AAAmodel',      # <--- 必须改这里！
        executable='tactical_ai',
        name='ai_engine',
        output='screen'
    )

    return LaunchDescription([
        vision_node,
        ai_node
    ])