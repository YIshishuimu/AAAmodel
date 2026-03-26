from ultralytics import YOLO
import os
import cv2
import numpy as np

def get_grid_index_from_mask(shell_mask, kfs_center):
    ys, xs = np.where(shell_mask > 0)
    if len(xs) == 0: return None
    s_x1, s_x2 = int(np.min(xs)), int(np.max(xs))
    s_y1, s_y2 = int(np.min(ys)), int(np.max(ys))
    shell_w = max(s_x2 - s_x1, 1)
    shell_h = max(s_y2 - s_y1, 1)
    kfs_cx, kfs_cy = kfs_center
    if not (s_x1 <= kfs_cx <= s_x2 and s_y1 <= kfs_cy <= s_y2): return None
    rel_x = (kfs_cx - s_x1) / shell_w
    rel_y = (kfs_cy - s_y1) / shell_h
    col = min(int(rel_x * 3), 2)
    row = min(int(rel_y * 3), 2)
    return row, col

def detect_color_in_mask(img_bgr, mask):
    if mask is None or np.sum(mask) == 0: return 0, "None"
    masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    )
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    r_count = cv2.countNonZero(mask_red)
    b_count = cv2.countNonZero(mask_blue)
    if r_count > b_count and r_count > 30: return 1, "Red"
    elif b_count > r_count and b_count > 30: return -1, "Blue"
    return 0, "Unknown"

def draw_matrix_overlay(img, matrix):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (180, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = (200, 200, 200)
            if val == 1: color = (0, 0, 255)
            if val == -1: color = (255, 0, 0)
            cv2.putText(img, str(val), (40 + j*45, 60 + i*35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("正在加载模型...")
    model_shell = YOLO(os.path.join(base_dir, "model", "shell_best-set.pt"))
    model_kfs = YOLO(os.path.join(base_dir, "model", "kfs_best-set.pt"))
    print("✅ 模型加载完成")

    input_dir = os.path.join(base_dir, "data", "test_images")
    output_root = os.path.join(base_dir, "data", "results")
    os.makedirs(output_root, exist_ok=True)

    print(f"正在推理 Shell 模型...")
    results_shell = model_shell.predict(source=input_dir, device="cpu", conf=0.3, imgsz=640, verbose=False)
    
    print(f"正在推理 KFS 模型...")
    results_kfs = model_kfs.predict(source=input_dir, device="cpu", conf=0.4, imgsz=640, verbose=False)

    print(f"开始后处理并保存...")
    
    for idx, (res_shell, res_kfs) in enumerate(zip(results_shell, results_kfs)):
        img_path = res_shell.path
        img_name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(results_shell)}] 处理: {img_name}")

        img = res_shell.orig_img
        h_img, w_img = img.shape[:2]
        draw_all = img.copy()
        matrix = np.zeros((3, 3), dtype=int)

        # --- 1. 处理 Shell (获取掩码) ---
        shell_mask_full = None
        if res_shell.masks is not None:
            shell_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            for i in range(len(res_shell.masks)):
                seg = res_shell.masks.xy[i]
                if len(seg) > 0:
                    pts = np.array(seg, dtype=np.int32)
                    cv2.fillPoly(shell_mask_full, [pts], 255)
                    cv2.drawContours(draw_all, [pts], -1, (0, 255, 0), 2)

        # --- 2. 适配 OBB 模型 处理 KFS ---
        # 核心修改：从 obb 里取旋转框，而不是 masks/boxes
        if hasattr(res_kfs, 'obb') and res_kfs.obb is not None and len(res_kfs.obb) > 0:
            for i in range(len(res_kfs.obb)):
                # 1. 从 OBB 里取旋转框的四个角点
                obb_item = res_kfs.obb[i]
                xyxyxyxy = obb_item.xyxyxyxy.cpu().numpy()[0]  # 四个角点坐标 (4,2)
                pts_kfs = xyxyxyxy.astype(np.int32)

                # 2. 计算中心点
                cx, cy = int(obb_item.xywhr[0][0]), int(obb_item.xywhr[0][1])

                # 3. 生成掩码用于颜色识别
                mask_kfs = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.fillPoly(mask_kfs, [pts_kfs], 255)

                # 4. 颜色识别
                color_val, color_name = detect_color_in_mask(img, mask_kfs)
                
                # 5. 确定画笔颜色
                bgr_color = (0, 255, 255)
                if color_val == 1: bgr_color = (0, 0, 255)
                if color_val == -1: bgr_color = (255, 0, 0)

                # 6. 九宫格逻辑
                pos_str = "Out"
                if shell_mask_full is not None:
                    grid_pos = get_grid_index_from_mask(shell_mask_full, (cx, cy))
                    if grid_pos:
                        r, c = grid_pos
                        matrix[r, c] = color_val
                        pos_str = f"[{r},{c}]"

                # 7. 绘制
                cv2.polylines(draw_all, [pts_kfs], True, bgr_color, 2)
                cv2.circle(draw_all, (cx, cy), 5, bgr_color, -1)
                cv2.putText(draw_all, f"{color_name} {pos_str}", (cx+10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

        # --- 保存 ---
        draw_all = draw_matrix_overlay(draw_all, matrix)
        out_path = os.path.join(output_root, f"final_{img_name}")
        cv2.imwrite(out_path, draw_all)

    print("🎉 全部处理完成！")

if __name__ == "__main__":
    main()