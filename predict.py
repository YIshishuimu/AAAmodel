import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge  # 用于 ROS 图像与 OpenCV 图像转换
from ultralytics import YOLO
import cv2
import numpy as np
import os

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        # --- 1. 参数与模型初始化 ---
        # 建议实际使用时通过参数服务器传递路径
        base_dir = "/your/absolute/path/to/models" 
        self.model_shell = YOLO(os.path.join(base_dir, "shell_best-set.pt"))
        self.model_kfs = YOLO(os.path.join(base_dir, "kfs_best-set.pt"))
        
        self.bridge = CvBridge()
        self.get_logger().info("✅ YOLO 模型已加载，节点启动")

        # --- 2. ROS 通信接口 ---
        # 订阅摄像头原始图像
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        # 发布处理后的标注图像（用于可视化调试）
        self.image_pub = self.create_publisher(Image, '/vision/result_image', 10)
        
        # 发布九宫格矩阵结果 (1x9 数组)
        self.matrix_pub = self.create_publisher(Int32MultiArray, '/vision/matrix', 10)

    # --- 原有逻辑复用 (Helper Functions) ---
    def get_grid_index_from_mask(self, shell_mask, kfs_center):
        ys, xs = np.where(shell_mask > 0)
        if len(xs) == 0: return None
        s_x1, s_x2 = int(np.min(xs)), int(np.max(xs))
        s_y1, s_y2 = int(np.min(ys)), int(np.max(ys))
        shell_w, shell_h = max(s_x2 - s_x1, 1), max(s_y2 - s_y1, 1)
        kfs_cx, kfs_cy = kfs_center
        if not (s_x1 <= kfs_cx <= s_x2 and s_y1 <= kfs_cy <= s_y2): return None
        rel_x, rel_y = (kfs_cx - s_x1) / shell_w, (kfs_cy - s_y1) / shell_h
        return min(int(rel_y * 3), 2), min(int(rel_x * 3), 2)

    def detect_color_in_mask(self, img_bgr, mask):
        if mask is None or np.sum(mask) == 0: return 0, "None"
        hsv = cv2.cvtColor(cv2.bitwise_and(img_bgr, img_bgr, mask=mask), cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])))
        mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        r_count, b_count = cv2.countNonZero(mask_red), cv2.countNonZero(mask_blue)
        if r_count > b_count and r_count > 30: return 1, "Red"
        if b_count > r_count and b_count > 30: return -1, "Blue"
        return 0, "Unknown"

    # --- 核心回调函数 ---
    def image_callback(self, msg):
        # 1. 将 ROS 图像消息转换为 OpenCV 格式
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        draw_all = frame.copy()
        matrix = np.zeros((3, 3), dtype=int)
        h_img, w_img = frame.shape[:2]

        # 2. 模型推理
        res_shell = self.model_shell.predict(frame, conf=0.3, verbose=False)[0]
        res_kfs = self.model_kfs.predict(frame, conf=0.4, verbose=False)[0]

        # 3. 处理 Shell 掩码
        shell_mask_full = None
        if res_shell.masks is not None:
            shell_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            for seg in res_shell.masks.xy:
                if len(seg) > 0:
                    cv2.fillPoly(shell_mask_full, [np.array(seg, dtype=np.int32)], 255)

        # 4. 处理 KFS (OBB)
        if hasattr(res_kfs, 'obb') and res_kfs.obb is not None:
            for obb_item in res_kfs.obb:
                pts_kfs = obb_item.xyxyxyxy.cpu().numpy()[0].astype(np.int32)
                cx, cy = int(obb_item.xywhr[0][0]), int(obb_item.xywhr[0][1])
                
                mask_kfs = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.fillPoly(mask_kfs, [pts_kfs], 255)
                color_val, _ = self.detect_color_in_mask(frame, mask_kfs)

                if shell_mask_full is not None:
                    grid_pos = self.get_grid_index_from_mask(shell_mask_full, (cx, cy))
                    if grid_pos:
                        matrix[grid_pos[0], grid_pos[1]] = color_val

                # 绘制代码 (省略部分绘图细节，保持简洁)
                cv2.polylines(draw_all, [pts_kfs], True, (0, 255, 255), 2)

        # 5. 发布矩阵数据
        mat_msg = Int32MultiArray()
        mat_msg.data = matrix.flatten().tolist()
        self.matrix_pub.publish(mat_msg)

        # 6. 发布可视化图像
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(draw_all, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()