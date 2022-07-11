#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import cv2
from cv2 import aruco
import glob
import os

def main():
    # マーカーサイズ
    marker_length = 0.132 # [m]
    # マーカーの辞書選択
    # dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# mtx[[2.98739863e+03 0.00000000e+00 1.61458343e+03]
#  [0.00000000e+00 2.99718990e+03 1.22753493e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# dist[ 8.02872253e-03  1.05598505e+00 -1.92094693e-03 -1.78528408e-03
#  -4.18652004e+00]
    camera_matrix = np.load("./mtx.npy")
    # camera_matrix = np.array([[2400.0, 0.00000000e+00, 1634.0], [0.00000000e+00, 2400.0, 1224.0], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distortion_coeff = np.load("./dist.npy")
    print("mtx:", camera_matrix, ", dist:", distortion_coeff)
    dir = "./ValImages/v2"
    images = sorted(glob.glob(dir + "/*"))
    images = reversed(images)
    # print(images)

    for i, image in enumerate(images):
        print(os.path.basename(image))
        img = cv2.imread(image)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
        # print(corners)
        # 可視化
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,255))

        if len(corners) == 0:
            print("Didn't Detect")
            
        else:
            
            # if len(corners) > 0:
                # マーカーごとに処理
                # for i, corner in enumerate(corners):
            # rvec -> rotation vector, tvec -> translation vector
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion_coeff)
    
            # < rodoriguesからeuluerへの変換 >
    
            # 不要なaxisを除去
            tvec = np.squeeze(tvec)
            rvec = np.squeeze(rvec)
            # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = cv2.Rodrigues(rvec)
            rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
            # 並進ベクトルの転置
            transpose_tvec = tvec[np.newaxis, :].T
            # 合成
            proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
            # オイラー角への変換
            euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
    
            # print("x : " + str(tvec[0]))
            # print("y : " + str(tvec[1]))
            # print("z : " + str(tvec[2]))
            # print("roll : " + str(euler_angle[0]))
            # print("pitch: " + str(euler_angle[1]))
            # print("yaw  : " + str(euler_angle[2]))
    
            # 可視化
            draw_pole_length = marker_length/2 # 現実での長さ[m]
            # aruco.drawAxis(img, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.putText(img, str(tvec), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img, "Cal: "+str('{:.4f}'.format(tvec[2]))+"[m]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "Acc:"+str(i+0.5)+"[m]", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img, "diff:"+str('{:.3f}'.format((tvec[2]-(i+0.5))*1000))+"[mm]", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
            # cv2.putText(img, str(euler_angle), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
            # cv2.putText(img, str(draw_pole_length), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            # print(image)
            cv2.imwrite("./Outputs/"+str(os.path.basename(image)), img)

if __name__ == '__main__':
    main()
