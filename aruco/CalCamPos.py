import cv2
import numpy as np
import glob
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_length = 0.132 #[m]
# mtx = np.array([[2400.0, 0.0, 1632.0], [0.0, 2400.0, 1224.0], [0.0, 0.0, 1.0]])
# dist = np.array([0, 0, 0, 0, 0])
mtx = np.load("./mtx.npy")
dist = np.load("./dist.npy")
# print(mtx, dist)

dir = "./ValImages"
images = glob.glob(dir + "/*")
print(images)
# start = 21
# num = 10
# path = "IMG_22.JPG"


for i in images:
    XYZ = []
    RPY = []
    V_x = []
    V_y = []
    V_z = []
    # path = 2221 + num
    print(i)
    img_bgr = cv2.imread(i)
    # print(dir+"/IMG_"+str(i)+".JPG")
    # print(img_bgr)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR2RGB
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
    
    if len(corners) == 0:
        print("Didn't Detect")
    
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
    
    R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
    R_T = R.T
    T = tvec[0].T
    # print(T)
    
    xyz = np.dot(R_T, - T).squeeze()
    XYZ.append(xyz)
    # print(XYZ)
    
    rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
    RPY.append(rpy)
    
    V_x.append(np.dot(R_T, np.array([1,0,0])))
    V_y.append(np.dot(R_T, np.array([0,1,0])))
    V_z.append(np.dot(R_T, np.array([0,0,1])))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB2BGR
    # ---- 描画
    cv2.aruco.drawDetectedMarkers(img, corners, ids, (0,255,255))
    # cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, marker_length/2)
    cv2.putText(img, str(XYZ), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite("./Outputs/"+str(os.path.basename(i)), img)
    # cv2.waitKey(1)
    # # ----
    
    # cv2.destroyAllWindows()