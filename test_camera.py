import cv2

import time


def take_photo():
    cv2.namedWindow("Resize Preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
        print('Original Dimensions : ', frame.shape)
    else:
        rval = False

    width = 640
    height = 480
    dim = (width, height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)

    while rval:
        cv2.imshow("Resize Preview", cv2.flip(frame, 1))
        rval, frame = vc.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        elif key == ord('s'):
            # 注意修改保持路径
            cv2.imwrite(
                '/Users/apple/PycharmProjects/face_BMI/data/height_weight_test/capture'
                + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                '.jpg', frame)
            cv2.imshow("capture", frame)

    cv2.destroyWindow("Resize Preview")
    cv2.destroyWindow("capture")
