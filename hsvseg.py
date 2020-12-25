# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
from numba import vectorize, jit, float64
import getopt
import sys

opts, args = getopt.getopt(sys.argv[1:], '-i:-o:')
for opt_name, opt_value in opts:
    if opt_name == '-i':
        input_path = opt_value
    if opt_name == '-o':
        output_path = opt_value

el = cv2.getTickCount()
cap = cv2.VideoCapture(input_path)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
width = int(cap.get(3))
height = int(cap.get(4))
pix = width * height
fps = cap.get(5)
tot_frame = cap.get(7)
print(tot_frame)  # 打印总帧数
pt = 0  # 指明目前子视频数
pt1 = 0  # 指明目前帧数
m = 0.7  # 相似度阈值
# L = [[], []]  # 用于储存相邻两帧的直方图
h_H = np.zeros(361, dtype=int)
s_S = np.zeros(11, dtype=int)
v_V = np.zeros(11, dtype=int)

# @vectorize(nopython=True)
@jit(nopython=True)
def zhifang(h: np.array, s: np.array, v: np.array):
    L = [0.] * 72
    h = h*2  # 0 -> 180 to 0 -> 360
    s = s/25.5
    v = v/25.5

    '''
    for i in range(0, height):
        for j in range(0, width):
            if ((h[i][j] >= 0) and (h[i][j] < 20)) or ((h[i][j] >= 316) and (h[i][j] < 360)):
                H = 0
            elif h[i][j] >= 20 and h[i][j] < 40:
                H = 1
            elif h[i][j] >= 40 and h[i][j] < 75:
                H = 2
            elif h[i][j] >= 75 and h[i][j] < 155:
                H = 3
            elif h[i][j] >= 155 and h[i][j] < 190:
                H = 4
            elif h[i][j] >= 190 and h[i][j] < 270:
                H = 5
            elif h[i][j] >= 270 and h[i][j] < 295:
                H = 6
            elif h[i][j] >= 295 and h[i][j] < 316:
                H = 7

            if s[i][j] >= 0 and s[i][j] < 0.2:
                S = 0
            elif s[i][j] >= 0.2 and s[i][j] < 0.7:
                S = 1
            elif s[i][j] >= 0.7 and s[i][j] <= 1:
                S = 2

            if v[i][j] >= 0 and v[i][j] < 0.2:
                V = 0
            if v[i][j] >= 0.2 and v[i][j] < 0.7:
                V = 1
            if v[i][j] >= 0.7 and v[i][j] <= 1:
                V = 2
            l = 9*H + 3*S + V
            # print(l)
            # print(L)
            L[l] += 1
    '''

    for i in range(height):
        for j in range(width):
            H = h_H[h[i][j]]
            S = s_S[int(s[i][j])]
            V = v_V[int(v[i][j])]
            l = 9*H + 3*S + V
            L[l] += 1
            #print(i, j)

    '''
    l = np.zeros((1080, 1920), dtype=int)
    for i in np.nditer(h):
        l += h_H[i]
    for i in np.nditer(s):
        l += s_S[int(i)]
    for i in np.nditer(v):
        l += v_V[int(i)]
    '''

    for i in range(0, 72):
        L[i] = L[i] / (height * width)

    '''
    debug_cnt = float(0)
    for i in range(72):
        debug_cnt += L[i]
    print(debug_cnt)
    '''

    return L

def similar(L1, L2):
    re = 0
    # print(L1[1])
    for i in range(0, 72):
        # print(L1[i])
        re += min(L1[i], L2[i])
    return re

ret, frame = cap.read()  # 预处理第一帧
if ret == True:

    # 预处理
    for i in range(361):
        if ((i >= 0) and (i < 20)) or ((i >= 316) and (i < 360)):
            h_H[i] = 0
        elif i >= 20 and i < 40:
            h_H[i] = 1
        elif i >= 40 and i < 75:
            h_H[i] = 2
        elif i >= 75 and i < 155:
            h_H[i] = 3
        elif i >= 155 and i < 190:
            h_H[i] = 4
        elif i >= 190 and i < 270:
            h_H[i] = 5
        elif i >= 270 and i < 295:
            h_H[i] = 6
        elif i >= 295 and i < 316:
            h_H[i] = 7
    for i in range(11):
        if i >= 0 and i < 2:
            s_S[i] = 0
            v_V[i] = 0
        elif i >= 2 and i < 7:
            s_S[i] = 1
            v_V[i] = 1
        elif i >= 7 and i <= 10:
            s_S[i] = 2
            v_V[i] = 2

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = np.empty([1080, 1920], dtype=int)
    s = np.empty([1080, 1920], dtype=int)
    v = np.empty([1080, 1920], dtype=int)
    h, s, v = cv2.split(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  # 还原该帧

    L = [[], []]
    L[pt1 % 2] = zhifang(h, s, v)
    pt1 += 1
    name = output_path + '\\shot_' + str(pt) + ".mp4"
    out = cv2.VideoWriter(name, fourcc, fps, (width, height), True)
    out.write(frame)
else:
    print("无法打开文件！")
    os.system("pause")
    exit()

while(pt1 < tot_frame):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h = np.empty([1080, 1920], dtype=int)
        s = np.empty([1080, 1920], dtype=int)
        v = np.empty([1080, 1920], dtype=int)
        h, s, v = cv2.split(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        L[pt1 % 2] = zhifang(h, s, v)
        pt1 += 1
        if similar(L[0], L[1]) < m:
            pt += 1
            # name = 'shot_' + str(pt) + ".mp4"
            name = output_path + '\\shot_' + str(pt) + ".mp4"
            # os.system('ffmpeg -i \"' + output_path + '\\shot_' + str(pt - 1) + '.mp4\" -b:v 1M \"' + output_path + '\\shot' + str(pt - 1) + '.mp4\"')
            # print('ffmpeg -i \'' + output_path + '\\shot_' + str(pt - 1) + '.mp4\' -b:v 1M \'' + output_path + '\\shot' + str(pt - 1) + '.mp4\'')
            out = cv2.VideoWriter(name, fourcc, fps, (width, height), True)
            out.write(frame)
            print("切分")
            print(pt1)
            # print(name)
        else:
            out.write(frame)
            print(pt1)
    '''
    if cv2.waitKey(1) == ord('q'):
        print("结束")
        break
    else:
        print("结束")
        break
    '''

cap.release()
out.release()
for i in range(pt + 1):
    os.system('ffmpeg -i \"' + output_path + '\\shot_' + str(i) + '.mp4\" -b:v 1M \"' + output_path + '\\shot-' + str(i) + '.mp4\"')
    os.system('rm \"' + output_path + '\\shot_' + str(i) + '.mp4\"')

# out.release()
e2 = cv2.getTickCount()
# print((e2.el)/cv2.getTickFrequency)
os.system("pause")
