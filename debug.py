from ultralytics import YOLO
import os
import cv2
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 只加载 KFS 模型
    print("正在加载 KFS 模型...")
    model_kfs = YOLO(os.path.join(base_dir, "model", "kfs_best-set.pt"))
    print("✅ KFS 模型加载成功")

    # 路径
    input_dir = os.path.join(base_dir, "data", "test_images")
    output_dir = os.path.join(base_dir, "data", "results", "kfs_only")
    os.makedirs(output_dir, exist_ok=True)

    # 推理
    print("开始推理 KFS...")
    results_kfs = model_kfs.predict(
        source=input_dir,
        device="cpu",
        conf=0.1,
        imgsz=640,
        verbose=False
    )

    # 逐张处理
    for res in results_kfs:
        img_name = os.path.basename(res.path)
        img = res.orig_img.copy()
        h, w = img.shape[:2]

        print(f"处理：{img_name}")

        # ==========================================
        # 核心：读取 KFS 原始标签（OBB 旋转框）
        # ==========================================
        if hasattr(res, 'obb') and res.obb is not None:
            for obb in res.obb:
                # 旋转框 4 个点
                pts = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                # 类别 ID（原始标签）
                cls_id = int(obb.cls.item())
                # 置信度
                conf = float(obb.conf.item())

                # 画旋转框（黄色）
                cv2.polylines(img, [pts], True, (0, 255, 255), 2)

                # 显示原始标签 + 置信度
                label = f"ID:{cls_id} ({conf:.2f})"
                cx, cy = np.mean(pts, axis=0).astype(int)
                cv2.putText(img, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)

        # 保存
        save_path = os.path.join(output_dir, f"kfs_{img_name}")
        cv2.imwrite(save_path, img)

    print(f"\n🎉 KFS 单独识别完成！结果保存在：\n{output_dir}")

if __name__ == "__main__":
    main()