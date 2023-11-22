import os
import re
from re import Match
from typing import Union

def get_window_info(window_classname: str) -> Union[tuple, None]:
    """
    根据窗口标题获取窗口信息
    params:
        window_classname (str): 窗口类名
    return: (左上角X坐标，左上角Y坐标，长，宽) | None
    """
    # 查找该窗口类名的ID列表
    window_ids: list = [window_id for window_id in os.popen(f"xdotool search --classname '{window_classname}'").read().split("\n") if window_id]
    # 如果ID列表为空则return
    if not window_ids:
        return None
    # 定义真实的ID
    real_window_id: str = ""   
    # 遍历ID列表，查找出能激活的窗口ID
    for window_id in window_ids:
        activate_result: str = os.popen(f"xdotool windowactivate {window_id}").read()
        # 如果激活窗口有输出则进行下一次循环
        if activate_result:
            continue
        # 如果激活窗口没有输出，赋值给real_window_id，说明是这个ID
        real_window_id = window_id
    # 如果real_window_id还是空字符串则return
    if not real_window_id:
        return None
    # 获取坐标和宽高
    window_info: list = os.popen(f"xdotool getwindowgeometry {real_window_id}").read().split("\n")[1:3]  # ['  Position: 957,182 (screen: 0)', '  Geometry: 800x666']    
    position_match: Match = re.search(r"  Position: (\d+,\d+) \(screen: \d+\)", window_info[0])  #  <re.Match object; span=(0, 31), match='  Position: 957,182 (screen: 0)'>
    size_match: Match = re.search(r"  Geometry: (\d+x\d+)", window_info[1])  # <re.Match object; span=(0, 19), match='  Geometry: 800x666'>    
    left: int = int(position_match.group(1).split(",")[0])
    top: int = int(position_match.group(1).split(",")[1])
    width: int = int(size_match.group(1).split("x")[0])
    height: int = int(size_match.group(1).split("x")[1])
    return left, top, width, height    