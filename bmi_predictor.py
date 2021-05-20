# data_folder = "height_weight"
#
# from glob import glob
# all_files = glob(data_folder+"/*")
#
# all_jpgs = sorted([img for img in all_files if ".jpg" in img or ".jpeg" in img or "JPG" in img])
#
# print("Total {} photos ".format(len(all_jpgs)))
#
# from pathlib import Path as p
#
# def get_index_of_digit(string):
#     import re
#     match = re.search("\d", p(string).stem)
#     return match.start(0)
#
# id_path = [(p(images).stem[:(get_index_of_digit(p(images).stem))],images) for  images in all_jpgs ]
#
# label_file = "data/BMI data - Sheet1.csv"
#
# import pandas as pd
#
# image_df = pd.DataFrame(id_path,columns=['UID','path'])
#
# profile_df = pd.read_csv(label_file)
#
# data_df = image_df.merge(profile_df)
#
import face_recognition
import numpy as np


def get_face_encoding(image_path):
    print(image_path)
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print("no face found !!!")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


#
# all_faces = []
#
# for images in data_df.path:
#     face_enc = get_face_encoding(images)
#     all_faces.append(face_enc)
#
# X = np.array(all_faces)
#
# y_height = data_df.height.values ## all labels
# y_weight = data_df.weight.values
# y_BMI = data_df.BMI.values
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_height_train, y_height_test, y_weight_train, y_weight_test ,y_BMI_train, y_BMI_test = train_test_split(X, y_height,y_weight,y_BMI, random_state=1)
#
#
# def report_goodness(model, X_test, y_test, predictor_log=True):
#     # Make predictions using the testing set
#     y_pred = model.predict(X_test)
#     y_true = y_test
#     if predictor_log:
#         y_true = np.log(y_test)
#     # The coefficients
#     # The mean squared error
#     print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % r2_score(y_true, y_pred))
#
#     errors = abs(y_pred - y_true)
#     mape = 100 * np.mean(errors / y_true)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
#
# from sklearn.kernel_ridge import KernelRidge
# from sklearn import  linear_model
# from sklearn.svm import SVR
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

import joblib

height_model = joblib.load('models/weight_predictor.model')
weight_model = joblib.load('models/height_predictor.model')
bmi_model = joblib.load('models/bmi_predictor.model')

height_model_simple_linear_regression = joblib.load('models/height_predictor_simple_linear_regression.model')
weight_model_simple_linear_regression = joblib.load('models/weight_predictor_simple_linear_regression.model')
bmi_model_simple_linear_regression = joblib.load('models/bmi_predictor_simple_linear_regression.model')

height_model_Ridge_Linear_Regression = joblib.load('models/height_predictor_Ridge_Linear_Regression.model')
weight_model_Ridge_Linear_Regression = joblib.load('models/weight_predictor_Ridge_Linear_Regression.model')
bmi_model_Ridge_Linear_Regression = joblib.load('models/bmi_predictor_Ridge_Linear_Regression.model')

height_model_Random_Forest_Regressor = joblib.load('models/height_predictor_Random_Forest_Regressor.model')
weight_model_Random_Forest_Regressor = joblib.load('models/weight_predictor_Random_Forest_Regressor')
bmi_model_Random_Forest_Regressor = joblib.load('models/bmi_predictor_Random_Forest_Regressor.model')


# 加载model

def predict_height_width_BMI(test_image, height_model, weight_model, bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
    height = np.asscalar(np.exp(height_model.predict(test_array)))
    weight = np.asscalar(np.exp(weight_model.predict(test_array)))
    bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
    return {'height': round(height, 2), 'weight': round(weight, 2), 'bmi': round(bmi, 2)}
