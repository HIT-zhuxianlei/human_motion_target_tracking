import cv2
import torch
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from config import model_path
from ultralytics.engine.results import Keypoints, Boxes
from utils import tensor_to_list

# 加载Yolo模型
model: YOLO = YOLO(model_path)
# 图片目录
source_dir: str = "images"

cap = cv2.VideoCapture(0)
# 检查是否启用CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
else:
    device = torch.device("cpu")  # 使用CPU
count = 10
while 1:
    # count = count - 1
    # if count == 0 :
    #     break
    print("start:")
    ret, frame = cap.read()
    result: Results = model.predict(source=frame, device=device)[0]

    keypoints: Keypoints = result.keypoints  # 关键点数据
    res_plotted = result.plot()
    cv2.imshow("result", res_plotted)
    boxes: Boxes = result.boxes
    print("boxes is")
    print(boxes)
    confidence_tensor: Tensor = keypoints.conf  # 关键点关联的置信度对象
    xy_tensor: Tensor = keypoints.xy  # 关键点对象
    if not all([xy_list := tensor_to_list(xy_tensor), confidence_list := tensor_to_list(confidence_tensor)]):
        continue

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
    print("boxes type is:", type(boxes))
    print("keypoints type is:", type(keypoints))
        
    cv2.imshow("test", frame)

    # 判断是否按下ESC
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cv2.destroyAllWindows()
