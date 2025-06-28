import os
import sys

# 请在此输入您的代码

# def lcm(a, b):
#     for i in range(min(a,b),0,-1):
#         if a % i ==0 and b % i == 0:
#             return a*b//i
#
#
#
# lst = [float('inf')]*(2021+1)
# lst[1] = 0 # 节点从1开始
# for i in range(1, 2022):  # 节点从1开始
#   for j in range(i + 1, i + 22):
#     if j > 2021:   # 跳出循环
#       break
#     lst[j] = min(lst[j], lst[i] + lcm(i, j))
# print(lst[2021])
#
# from ultralytics import YOLO
# import cv2
# import numpy as np
#
# # Load a pretrained YOLOv8n model
# model = YOLO('./test_model/wiou.pt')
#
# # Run inference on 'bus.jpg' with arguments
# model.predict('./test_images/2.jpg', save=True, save_txt=True)
#
#


import cv2
import numpy as np

# 读取图像
image = cv2.imread('F:/yolov8/ultralytics/test_images/2.jpg')

w = 640
h = 640

a,b,c,d = 0.517655, 0.569754, 0.268066, 0.368599

# 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]  # 示例检测结果

# 绘制矩形框和在框内添加浅色
for detection in detections:
    x_min, y_min, x_max, y_max, class_id = detection
    color = (255, 255, 0)  # 框的颜色，这里假设是黄色
    thickness = 1  # 框的厚度
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # 在框内添加浅色
    alpha = 0.2  # 透明度E
    overlay = image.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# 显示结果图像
#
a,b,c,d = 0.456924 ,0.477268, 0.150669 ,0.183733
# 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]  # 示例检测结果

# 绘制矩形框和在框内添加浅色
for detection in detections:
    x_min, y_min, x_max, y_max, class_id = detection
    color = (255, 0, 0)  # 框的颜色，这里假设是黄色
    thickness = 1  # 框的厚度
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # 在框内添加浅色
    alpha = 0.2  # 透明度E
    overlay = image.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# a,b,c,d = 0.579217, 0.566395, 0.14956, 0.36175
# # 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
# detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]  # 示例检测结果
#
# # 绘制矩形框和在框内添加浅色
# for detection in detections:
#     x_min, y_min, x_max, y_max, class_id = detection
#     color = (0, 255, 0)  # 框的颜色，这里假设是黄色
#     thickness = 1  # 框的厚度
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
#
#     # 在框内添加浅色
#     alpha = 0.2  # 透明度E
#     overlay = image.copy()
#     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
#     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#
#
# a,b,c,d =0.469261 ,0.561654, 0.179365, 0.352384
# # 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
# detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]  # 示例检测结果
#
# # 绘制矩形框和在框内添加浅色
# for detection in detections:
#     x_min, y_min, x_max, y_max, class_id = detection
#     color = (0, 0, 255)  # 框的颜色，这里假设是黄色
#     thickness = 1  # 框的厚度
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
#
#     # 在框内添加浅色
#     alpha = 0.2  # 透明度E
#     overlay = image.copy()
#     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
#     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#
# a,b,c,d = 0.620464 ,0.596279 ,0.212374, 0.639407
# # 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
# detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]   # 示例检测结果
#
# # 绘制矩形框和在框内添加浅色
# for detection in detections:
#     x_min, y_min, x_max, y_max, class_id = detection
#     color = (0, 255, 255)  # 框的颜色，这里假设是黄色
#     thickness = 1  # 框的厚度
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
#
#     # 在框内添加浅色
#     alpha = 0.2  # 透明度E
#     overlay = image.copy()
#     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
#     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#
# a,b,c,d = 00.530603, 0.529531, 0.289809, 0.529386
# # 假设detections是从YOLO模型得到的目标检测结果，格式为[x_min, y_min, x_max, y_max, class_id]
# detections = [[int(a*w)-int(c*w/2), int(b*h)-int(d*h/2),int(a*w)+int(c *w/2), int(b*h)+int(d*h/2), 0]]  # 示例检测结果
#
# # 绘制矩形框和在框内添加浅色
# for detection in detections:
#     x_min, y_min, x_max, y_max, class_id = detection
#     color = (255, 0, 255)  # 框的颜色，这里假设是黄色
#     thickness = 1  # 框的厚度
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
#
#     # 在框内添加浅色
#     alpha = 0.2  # 透明度E
#     overlay = image.copy()
#     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
#     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


cv2.imwrite('wiou-2.jpg', image)
cv2.imshow('images',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
