首先确认视频流地址正确，例如：



WINDOWS\_IP = "172.23.16.1"

stream\_url = f"http://{WINDOWS\_IP}:8080/video.mjpg"



然后运行：



python main.py



运行后会打开两个窗口：



WSL YOLO Inspection System：YOLO 检测结果窗口

Cropped ROI：仪表盘 ROI 指针识别调试窗口



按下 q 退出程序。



核心算法说明

1\. YOLO 定位仪表盘



YOLO 负责在整张图像中检测仪表盘位置，输出目标框坐标：



results = model.predict(source=frame, conf=trust, verbose=False)

2\. ROI 裁剪



根据 YOLO 检测框截取仪表盘区域：



roi\_image = frame\[y1:y2, x1:x2].copy()

3\. 指针提取



对 ROI 图像进行灰度化和二值化，提取黑色指针区域：



gray = cv2.cvtColor(roi, cv2.COLOR\_BGR2GRAY)

\_, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH\_BINARY\_INV)



同时使用圆形掩膜减少表盘外部干扰。



4\. 指针尖端检测



在有效轮廓中寻找距离圆心最远的点，认为该点为指针尖端。



5\. HSV 区域判断



沿着指针方向延长采样点，读取该点的 HSV 色相值，根据颜色判断状态：



if (0 <= hue <= 14) or (160 <= hue <= 179):

&#x20;   return "偏高"

elif 15 <= hue <= 35:

&#x20;   return "偏低"

elif 35 < hue <= 85:

&#x20;   return "正常"

else:

&#x20;   return "未知区域"

6\. 滑动窗口滤波



为了减少单帧识别抖动，程序使用一个长度为 10 的队列保存最近识别结果，并取众数作为最终输出：



history\_buffer = deque(maxlen=10)

final\_status = Counter(history\_buffer).most\_common(1)\[0]\[0]

项目特点



本项目不是单纯调用 YOLO 进行分类，而是采用了“YOLO 定位 + OpenCV 几何分析 + HSV 颜色判断”的混合方案。



YOLO 负责解决“仪表盘在哪里”的问题，OpenCV 负责解决“指针指向哪里”的问题。这样可以减少对大量分类数据集的依赖，也方便根据实际表盘颜色和结构进行参数调整。



后续优化方向

增加更多仪表盘样本，提高 YOLO 检测稳定性

将 HSV 阈值改为配置文件，方便现场调参

增加异常状态报警功能

增加识别结果保存和日志记录

支持本地摄像头、RTSP、MJPG 等多种视频输入方式

将识别结果发送给上位机、机器人或嵌入式控制系统

