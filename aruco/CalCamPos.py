import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_length = 0.132 #[m]
mtx = np.array([[2400.0, 0.0, 1632.0], [0.0, 2400.0, 1224.0], [0.0, 0.0, 1.0]])
dist = np.array([0, 0, 0, 0, 0])
# print(mtx, dist)

dir = "./ValImages/"
path = "IMG_2220.JPG"

XYZ = []
RPY = []
V_x = []
V_y = []
V_z = []


img_bgr = cv2.imread(dir+path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR2RGB
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)

if len(corners) == 0:
    print("Didn't Detect")

rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
R_T = R.T
T = tvec[0].T
print(T)

xyz = np.dot(R_T, - T).squeeze()
XYZ.append(xyz)
print(XYZ)

rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
RPY.append(rpy)

V_x.append(np.dot(R_T, np.array([1,0,0])))
V_y.append(np.dot(R_T, np.array([0,1,0])))
V_z.append(np.dot(R_T, np.array([0,0,1])))

# ---- 描画
cv2.aruco.drawDetectedMarkers(img, corners, ids, (0,255,255))
# cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, marker_length/2)
cv2.imwrite(dir+"af_"+path, img)
# cv2.waitKey(1)
# # ----

# cv2.destroyAllWindows()