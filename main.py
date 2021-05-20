import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import bmi_predictor
import cv2


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.file_label = QLabel(self)
        self.initUI()

    def loadFile(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '选择图片',
                                                    '/Users/apple/PycharmProjects/face_BMI/data/height_weight_test',
                                                    'Image files(*.jpg *.gif *.png)')
        print(self.fname)
        self.file_label.setPixmap(QPixmap(self.fname))
        self.file_label.setScaledContents(True)

    # 接受预测值
    def onChanged(self):
        # 保存返回的身高体重BMI字典
        dict_bmi = bmi_predictor.predict_height_width_BMI(self.fname, bmi_predictor.height_model,
                                                          bmi_predictor.weight_model,
                                                          bmi_predictor.bmi_model)
        # 提取bmi值
        bmi = dict_bmi['bmi']

        text = str(bmi_predictor.predict_height_width_BMI(self.fname, bmi_predictor.height_model,
                                                          bmi_predictor.weight_model,
                                                          bmi_predictor.bmi_model))

        # 将文本添加健康建议
        if bmi < 18.5:
            text = text + "您的BMI: 偏瘦 \n " \
                          "健康建议：\n 多吃含蛋白质高脂肪等高营养物质,可以多吃牛奶,牛肉,鸡蛋等," \
                          "尽量选择早晚上服用可以加速脂肪的堆积,应该有足够的睡觉," \
                          "吃夜宵多为没有营养刺激性的食物应该少吃,不能增加肠道功能的负担," \
                          "营养主要靠肠道来吸收的," \
                          "\n 多参加锻炼,多做引体向上,俯卧撑等活动增强肌肉锻炼 "

        elif bmi < 24:
            text = text + "您的BMI： 正常 \n" \
                          "健康建议：\n 恭喜，您非常健康"

        elif bmi < 27.5:
            text = text + "您的BMI： 偏重 \n" \
                          "健康建议：\n 首先要合理安排你的一日三餐,多吃蔬菜水果,早上吃好,中午吃饱,晚上吃少" \
                          "\n 早上起来先喝一杯温开水或蜂蜜水,肉要放在中午或早上吃." \
                          "\n 运动方面呢,可以在早晚出去跑跑步,跳跳绳,做做仰卧起坐,跳跳舞等等."

        else:
            text = text + "您的BMI： 肥胖 \n"\
                          "健康建议： \n 平时就应该注意清淡的饮食，少吃油腻的食物，" \
                          "以及含糖量高的食物，像动物脂肪，油炸食品，巧克力，蛋糕，饮料等就应该少吃了。" \
                          "\n 还应该控制饮食的量，每餐都不要吃太多，尤其是晚餐更应该少吃，" \
                          "吃一些热量低饱腹感强的食物，而且晚餐后一小时最好是去运动。"

        self.prediction_label.setText(text)
        self.prediction_label.adjustSize()

    def onChangedSimpleLiner(self):
        text = str(
            bmi_predictor.predict_height_width_BMI(self.fname, bmi_predictor.weight_model_simple_linear_regression,
                                                   bmi_predictor.height_model_simple_linear_regression,
                                                   bmi_predictor.bmi_model_simple_linear_regression))
        self.prediction_label.setText(text)
        self.prediction_label.adjustSize()

    def onChangedRidgeLinearRegression(self):
        text = str(
            bmi_predictor.predict_height_width_BMI(self.fname, bmi_predictor.weight_model_Ridge_Linear_Regression,
                                                   bmi_predictor.height_model_Ridge_Linear_Regression,
                                                   bmi_predictor.bmi_model_Ridge_Linear_Regression))
        self.prediction_label.setText(text)
        self.prediction_label.adjustSize()

    def onChangedRandomForestRegressor(self):
        text = str(
            bmi_predictor.predict_height_width_BMI(self.fname, bmi_predictor.weight_model_Random_Forest_Regressor,
                                                   bmi_predictor.height_model_Random_Forest_Regressor,
                                                   bmi_predictor.bmi_model_Random_Forest_Regressor))
        self.prediction_label.setText(text)
        self.prediction_label.adjustSize()

    def snapShotCt(self):  # camera_idx的作用是选择摄像头。如果为0则使用内置摄像头，比如笔记本的摄像头，用1或其他的就是切换摄像头。
        """拍照函数"""
        camera_idx = 0
        cap = cv2.VideoCapture(camera_idx)
        ret, frame = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
        cv2.imwrite("/Users/apple/PycharmProjects/face_BMI/data/height_weight_test/capture.jpg", frame)  # 写入图片
        cap.release()  # 释放




    def initUI(self):
        # 选择的图片
        self.file_label.move(300, 50)
        self.file_label.resize(500, 400)
        self.file_label.setObjectName("file_label")

        # 选择文件按钮
        pushButton_file_label = QPushButton("选择文件", self)
        pushButton_file_label.move(30, 50)
        pushButton_file_label.setObjectName("pushButton_file_label")
        pushButton_file_label.clicked.connect(self.loadFile)

        # 拍照按钮
        pushButton_snapshot = QPushButton("拍照", self)
        pushButton_snapshot.move(30, 100)
        pushButton_snapshot.setObjectName("pushButton_snapshot")
        pushButton_snapshot.clicked.connect(self.snapShotCt)

        # 显示预测结果标签
        self.prediction_label = QLabel(self)
        self.prediction_label.move(180, 600)

        # 预测按钮
        prediction_pushButton = QPushButton("开始预测", self)
        prediction_pushButton.move(150, 50)
        prediction_pushButton.clicked.connect(self.onChanged)

        # 预测按钮simple_linear_regression
        prediction_pushButton = QPushButton("简单线性", self)
        prediction_pushButton.move(150, 100)
        prediction_pushButton.clicked.connect(self.onChangedSimpleLiner)

        # 预测按钮Ridge Linear Regression
        prediction_pushButton = QPushButton("岭回归", self)
        prediction_pushButton.move(150, 150)
        prediction_pushButton.clicked.connect(self.onChangedRidgeLinearRegression)

        # 预测按钮Random Forest Regressor
        prediction_pushButton = QPushButton("随机森林", self)
        prediction_pushButton.move(150, 200)
        prediction_pushButton.clicked.connect(self.onChangedRandomForestRegressor)

        self.setGeometry(300, 300, 850, 800)
        self.setWindowTitle('预测人体BMI')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    with open('MacOS.qss', 'r', encoding='UTF-8') as f:
        app.setStyleSheet(f.read())
    ex = Example()
    sys.exit(app.exec_())
