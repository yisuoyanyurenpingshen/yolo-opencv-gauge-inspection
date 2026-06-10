import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque, Counter # [优化] 引入双端队列和计数器

model = YOLO("yol0+cv/yolo and cv/weights/best.pt")
trust = 0.5
# 视频的借调，这里就是调用cv2的意义
WINDOWS_IP = "172.23.16.1"  
stream_url = f"http://{WINDOWS_IP}:8080/video.mjpg"
print(f"🔗 正在尝试连接 Windows 视频流: {stream_url} ...")
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ 无法连接！请检查 win_cam.py 是否在运行，以及 IP 地址是否正确。")
    exit()
print("✅ 成功连接！算法模块启动，开始巡检...")

# [优化] 初始化一个最大长度为 10 的 FIFO (先进先出) 双端队列
# 它相当于我们的“滑动记忆窗口”，最多只记住最近 10 次的识别结果
history_buffer = deque(maxlen=10)

# 【工具区】定义 CV 底层算法
def get_pointer_status(roi):
    h, w = roi.shape[:2]
    # 假设 YOLO 裁下来的框，正中心就是表盘的圆心
    center_x, center_y = w // 2, h // 2

    # A. 找黑色指针 (灰度图 + 二值化)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 🔥 终极手术：圆形掩膜 (Mask) 降维打击
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    radius = int(min(w, h) / 2 * 0.6)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # B. 寻找所有轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "未能识别到任何黑色块"

    # C. 不要盲目找最大面积！真正的指针必须“连接着圆心”
    valid_contours = []
    for cnt in contours:
        dist = cv2.pointPolygonTest(cnt, (center_x, center_y), True)
        if dist >= -25: 
            valid_contours.append(cnt)

    if not valid_contours:
        return "未找到连接圆心的指针"

    # C2. 最终确认：在所有挨着圆心的黑块里，挑出面积最大的
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # D. 找指针尖端
    max_dist = 0
    tip_x, tip_y = center_x, center_y
    for point in largest_contour:
        px, py = point[0]
        dist = math.hypot(px - center_x, py - center_y)
        if dist > max_dist:
            max_dist = dist
            tip_x, tip_y = px, py

    # E. 几何射线采样
    extend_ratio = 1.35  
    sample_x = int(center_x + (tip_x - center_x) * extend_ratio)
    sample_y = int(center_y + (tip_y - center_y) * extend_ratio)

    # 防止采样点越过图片边界引发数组越界崩溃
    sample_x = max(0, min(sample_x, w - 1))
    sample_y = max(0, min(sample_y, h - 1))

    # ---------------- 必须先提取颜色！ ----------------
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixel_hsv = hsv_roi[sample_y, sample_x]
    hue = pixel_hsv[0] 

    # ---------------- 可视化调试 ----------------
    cv2.line(roi, (center_x, center_y), (tip_x, tip_y), (0, 255, 255), 2) 
    cv2.circle(roi, (sample_x, sample_y), 5, (255, 0, 0), -1)             

    # [优化] 修复了 11-14 之间的盲区断层，将其划分给“偏高”或“偏低”都可以，这里我划给了“偏高”(红色/橙红色)
    if (0 <= hue <= 14) or (160 <= hue <= 179):
        return "偏高"
    elif 15 <= hue <= 35:
        return "偏低"
    elif 35 < hue <= 85:
        return "正常"
    else:
        return "未知区域"
# 【执行区】主循环 
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 视频流读取失败或中断")
        break

    # 1. YOLO 仅负责定位表盘
    results = model.predict(source=frame, conf=trust, verbose=False)
    
    # 2. 遍历找到的表盘
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi_image = frame[y1:y2, x1:x2].copy() 

        # 3. 调用 CV 算法获取单帧状态
        current_status = get_pointer_status(roi_image)

        # 4. [优化] 将当前状态存入历史队列
        history_buffer.append(current_status)

        # 5. [优化] 核心逻辑：众数滤波计算最终状态
        # 解决“冷启动”问题：如果刚开机，队列里只有 1-2 帧，我们依然选出现次数最多的
        # Counter().most_common(1) 返回类似 [('正常', 8)] 的格式，[0][0] 提取出字符串 '正常'
        final_status = Counter(history_buffer).most_common(1)[0][0]

        # 6. 打印底层识别结果 vs 滤波后的最终结果，方便你观察滤波效果！
        print(f"目前识别的是{final_status} ")

        # 7. 🔥 完全满足省赛要求的终端输出
        # [优化] 严格对齐了 if-elif-else 的缩进，防止误报！
        if final_status == "正常":
            print("仪表盘正常")
        elif final_status == "偏高":
            print("仪表盘偏高，状态异常")
        elif final_status == "偏低":
            print("仪表盘偏低，状态异常")
        else:
            print("⚠️ 无法输出赛题要求结果，请看上面的 Debug 信息调整 CV 参数！")
        
        cv2.imshow("Cropped ROI", roi_image)

    annotated_frame = results[0].plot()
    cv2.imshow("WSL YOLO Inspection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()