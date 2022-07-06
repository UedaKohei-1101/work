import cv2
from cv2 import aruco
import os

### --- parameter --- ###

# マーカーの保存先
dir_mark = "./Markers/"

# 生成するマーカー用のパラメータ
num_mark = 9 #個数
size_mark = 500 #マーカーのサイズ

### --- マーカーを生成して保存する --- ###
# マーカー種類を呼び出し
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
# dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# marker = aruco.drawMarker(dict_aruco, 4, 100)
# cv2.imwrite("./Markers/mark.png", marker)

for count in range(num_mark) :
    # print(count)
    id_mark = count #countをidとして流用
    img_mark = aruco.drawMarker(dict_aruco, id_mark, size_mark)

    if count < 10 :
        img_name_mark = 'id_0' + str(count) + '.png'
    else :
        img_name_mark = 'id_' + str(count) + '.png'
    path_mark = os.path.join(dir_mark, img_name_mark)
    print(path_mark)
    cv2.imwrite(path_mark, img_mark)
    
# # print("done")