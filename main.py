# coding: utf-8
# @Author:Afun
# 用来识别视频并保存
import paddle
import paddle.fluid as fluid
import numpy as np
import cv2


def get_img(face_roi):
    img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = np.asarray(img, dtype='float32')
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# 加载模型和标签
model_save_dir = './model/face'
name_dict = {'keqin': 0, 'leisheng': 1, 'nilu': 2, 'vinti': 3, 'zhongli': 4}
name_list = ['Keqing', 'Raiden Shogun ', 'Nilou', 'Venti', 'Zhongli']

# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
paddle.enable_static()

# 加载预测程序
infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(model_save_dir, infer_exe)

# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

# 加载视频文件
cap = cv2.VideoCapture('test.mp4')

# 设置保存视频的相关参数
output_file = 'output.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    # 读取视频帧
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break

    # 转换成灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.18, minNeighbors=5, minSize=(40, 40))

    # 对每一个检测到的人脸进行预测和标注
    for (x, y, w, h) in faces:
        # 提取人脸ROI
        if y - 60 < 0:
            continue
        face_roi = frame[y - 60:y + h, x:x + w]
        # 图像预处理
        img = get_img(face_roi)
        # 执行预测
        results = infer_exe.run(infer_program,
                                feed={feed_target_names[0]: img},
                                fetch_list=fetch_targets)
        # 获取预测结果
        result = np.argmax(results[0])
        label = name_list[result]
        # 绘制人脸框和预测结果
        cv2.rectangle(frame, (x, y - 60), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 将帧写入输出视频文件
    out.write(frame)

    # 显示帧
    cv2.imshow('Face Recognition', frame)

    # 按下Esc键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头、关闭视频保存和销毁窗口
cap.release()
out.release()
cv2.destroyAllWindows()
