import cv2
import numpy as np
import torch
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from config import model_path
from ultralytics.engine.results import Keypoints, Boxes
from utils import tensor_to_list

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# 加载Yolo模型
model: YOLO = YOLO(model_path)
# 图片目录
source_dir: str = "images"


# 初始化DeepSORT追踪器
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


cap = cv2.VideoCapture(0)
# 检查是否启用CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
else:
    device = torch.device("cpu")  # 使用CPU
count = 10
while 1:
    print("#########################start:")
    ret, frame = cap.read()
    result: Results = model.predict(source=frame, device=device)[0]

    keypoints: Keypoints = result.keypoints  # 关键点数据
    res_plotted = result.plot()
    cv2.imshow("result", res_plotted)
    yolo_boxes: Boxes = result.boxes
    # print("boxes is")
    # print(yolo_boxes)
    
    
    # 准备deepsort输入数据
    boxes = []
    confidence = 0.9
    if confidence > 0.5 and len(yolo_boxes.xywh.cpu()): 
        for i in range(1):
            boxes.append(yolo_boxes.xywh.cpu().numpy()[i])
            
    confidence_tensor: Tensor = keypoints.conf  # 关键点关联的置信度对象
    xy_tensor: Tensor = keypoints.xy  # 关键点对象
    if not all([xy_list := tensor_to_list(xy_tensor), confidence_list := tensor_to_list(confidence_tensor)]):
        continue

    # 使用DeepSORT进行人体追踪
    for box in boxes:
        # 提取特征向量并进行追踪
        detections = [Detection(box, 1.0, feature=box)]
        tracker.predict()
        tracker.update(detections)

        # 在图像上绘制追踪结果
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # 获取追踪目标的边框坐标
            class_name = "deep_sort person"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 2)


    # 将所有头部关键点以及置信度保存到head_data_list，格式为[(1, 1), 0.98983]
    head_data_list: list[tuple[int], float] = []
    for xy, confidence in zip(xy_list, confidence_list):
        x: int = int(xy[0][0])
        y: int = int(xy[0][1])
        head_data_list.append([(x, y), confidence[0]])
        print("x is:", x, "    y is:", y, "\n")

    # 置信度最高的头部关键点以及置信度数据
    max_head_data: list[tuple[int], float] = max(head_data_list, key=lambda x: x[1])

    # 如果置信度小于0.7则跳过
    if max_head_data[1] < 0.7:
        continue

    # 读取图片将鼠标移动到图片中头部关键点位置
    center: tuple = max_head_data[0]  # 点的中心坐标
    radius: int = 3  # 点的半径
    color: tuple = (136, 53, 209)  # 点的颜色，这里使用紫色
    thickness: int = -1  # 填充整个圆形
    cv2.circle(frame, center, radius, color, thickness)
    print("boxes type is:", type(yolo_boxes))
    print("keypoints type is:", type(keypoints))
        
    cv2.imshow("test", frame)

    # 判断是否按下ESC
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cv2.destroyAllWindows()
