import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import numpy as np

# 假设你的项目结构中可以这样导入（或者把类定义直接放进来）
from AAAmodel.game import TacticalTicTacToe
from AAAmodel.AlphaBataBot import AlphaBetaBot # 确保类名拼写正确

class SimpleTacticalAiNode(Node):
    def __init__(self):
        super().__init__('simple_tactical_ai_node')
        
        # 1. 订阅视觉节点发布的矩阵 (1x9 整数数组)
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/vision/matrix',
            self.matrix_callback,
            10)
        
        # 2. 初始化 AI 和 环境
        self.env = TacticalTicTacToe()
        self.bot = AlphaBetaBot(depth=8)  # 深度可根据计算性能调整
        
        # 3. 武器赋值 (硬编码)
        self.default_weapons = {1: 3, -1: 3} 
        
        self.get_logger().info("🚀 决策节点已启动，监听 /vision/matrix...")

    def matrix_callback(self, msg):
        """核心逻辑：每当收到矩阵，就计算一次最佳步法"""
        
        # 将接收到的 1x9 数组转回 3x3 棋盘
        if len(msg.data) != 9:
            self.get_logger().error(f"收到错误的矩阵长度: {len(msg.data)}")
            return

        new_board = np.array(msg.data, dtype=np.int8).reshape((3, 3))
        
        # 只有当棋盘发生变化时才计算（可选，防止重复计算同一帧）
        # if np.array_equal(new_board, self.env.board): return

        # 更新环境状态
        self.env.board = new_board
        self.env.weapons = self.default_weapons.copy()
        self.env.current_player = 1  # 假设你是玩家 1 (○)
        
        self.get_logger().info("收到新棋盘，开始 AI 推理...")

        # 4. 调用 AI 算法
        try:
            best_action = self.bot.get_best_move(self.env)
            
            if best_action:
                self.get_logger().info("========================================")
                self.get_logger().info(f"💡 AI 建议操作: {best_action['type']}")
                self.get_logger().info(f"📍 目标位置: {best_action['pos']}")
                self.get_logger().info("========================================")
            else:
                self.get_logger().warn("AI 未能找到合法移动。")
                
        except Exception as e:
            self.get_logger().error(f"AI 计算过程中出错: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleTacticalAiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()