#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023-08-12 23:33
# @Author  : 陈伟峰
# @Site    : 
# @File    : MCIL_EM_CAL.py
# @Software: PyCharm
import numpy as np

def MCIL_EM_CAL(f,Alfa1,NN,L_spacing,T_depth,zi1,Thick,Rh1,Rv1,H_matrix,epsh1,epsv1):
    freq = f
    omega = 2 * np.pi * freq
    L = L_spacing
    NL = NN
    Alfa = Alfa1

    if Alfa <= 0.5:
        Alfa = 0.1
    elif Alfa >= 180:
        Alfa = 179.5
    elif np.abs(Alfa - 90) <= 0.5:
        Alfa = 89.5
    # 将alfa转换为弧度制
    Alfa = Alfa * np.pi / 180.

    thickness = Thick
    Rh = Rh1
    Rv = Rv1
    H_matrix = np.zeros((3, 3), dtype=np.complex128)
    # print("NL:",NL)

    print("thinkness:",thickness)
    zi_l = np.zeros(NL+1)
    # print("zi_l:",zi_l)
    zi_l[1] = zi1
    for i in range(2,NL):
        zi_l[i] = zi_l[i-1] + thickness[i-1]

    zi_l[0]= zi_l[1] - 1
    zi_l[NL] = zi_l[NL-1] + 1
    print("zi_l:",zi_l)
    exit()


    '''
    这两个参数x_t和z_t是用来表示目标点的水平和垂直位置的。
    在代码中，x_t表示目标点相对于原点的水平位置，z_t表示目标点相对于原点的垂直位置。
    在这段代码中，x_t被设置为0，表示目标点位于原点的水平位置，z_t被设置为T_depth，表示目标点位于T_depth的垂直位置。
    '''
    x_t = 0
    z_t = T_depth

    if Alfa <= np.pi / 2.:
        x_r = L * np.sin(Alfa)
        z_r = T_depth + L * np.cos(Alfa)
    else:
        Alfa = np.pi - Alfa
        x_r = L * np.sin(Alfa)
        z_r = T_depth - L * np.cos(Alfa)

    if (L > 0 and Alfa1 > 90.01) or (L < 0 and Alfa1 <= 90.01):
        Rh_l = np.flip(Rh)
        Rv_l = np.flip(Rv)
        zi_l1 = -np.flip(zi_l[1:])
        epsh_l = np.flip(epsh1)
        epsv_l = np.flip(epsv1)
        zi_l1 = np.append(zi_l1, -zi_l[0])
        z_t1 = -z_t
        z_r1 = -z_r
        if L > 0:
            x_t1 = x_t
            x_r1 = x_r
        else:
            x_t1 = -x_t
            x_r1 = -x_r
    else:
        Rh_l = Rh
        Rv_l = Rv
        zi_l1 = zi_l
        epsh_l = epsh1
        epsv_l = epsv1
        z_t1 = z_t
        z_r1 = z_r
        if L > 0:
            x_t1 = x_t
            x_r1 = x_r
        else:
            x_t1 = -x_t
            x_r1 = -x_r

    Z_T_R = z_r1 - z_t1
    X_T_R = np.abs(x_r1 - x_t1)
    sigmah_l = 1.0 / Rh_l - 1j * omega * epsh_l * eps0
    sigmav_l = 1.0 / Rv_l - 1j * omega * epsv_l * eps0
    lambda_l = np.sqrt(sigmah_l / sigmav_l)

    zi_t = zi_l1 - z_t1
    if zi_t[0] >= 0.:
        N_t = 1
    elif zi_t[NL-1] <= 0.:
        N_t = NL
    else:
        for j in range(NL-1):
            if zi_t[j] < 0. and zi_t[j+1] >= 0.:
                N_t = j + 1

    if zi_t[NL] <= Z_T_R:
        zi_t[NL] = zi_t[NL] + Z_T_R + 1

    if zi_t[0] >= Z_T_R:
        N_r = 1
    elif zi_t[NL-1] <= Z_T_R:
        N_r = NL
    else:
        for j in range(NL-1):
            if zi_t[j] < Z_T_R and zi_t[j+1] >= Z_T_R:
                N_r = j + 1

    Mag_solve()
    Alfa = np.abs(Alfa)
    H = np.zeros((3, 3), dtype=np.complex128)
    Alfa_mat = np.zeros((3, 3), dtype=np.complex128)
    H[0, 0] = Hxx_l
    H[0, 2] = Hxz_l
    H[1, 1] = Hyy_l
    H[2, 0] = Hzx_l
    H[2, 2] = Hzz_l

    if L > 0.:
        if Alfa1 < 90.01:
            Alfa_mat[0, 0] = np.cos(Alfa)
            Alfa_mat[0, 2] = np.sin(Alfa)
            Alfa_mat[2, 0] = -np.sin(Alfa)
            Alfa_mat[2, 2] = np.cos(Alfa)
            Alfa_mat[1, 1] = -1
        else:
            Alfa_mat[0, 0] = -np.cos(Alfa)
            Alfa_mat[0, 2] = np.sin(Alfa)
            Alfa_mat[2, 0] = np.sin(Alfa)
            Alfa_mat[2, 2] = np.cos(Alfa)
            Alfa_mat[1, 1] = 1
    else:
        if Alfa1 < 90.01:
            Alfa_mat[0, 0] = -np.cos(Alfa)
            Alfa_mat[0, 2] = -np.sin(Alfa)
            Alfa_mat[2, 0] = np.sin(Alfa)
            Alfa_mat[2, 2] = -np.cos(Alfa)
            Alfa_mat[1, 1] = -1
        else:
            Alfa_mat[0, 0] = np.cos(Alfa)
            Alfa_mat[0, 2] = -np.sin(Alfa)
            Alfa_mat[2, 0] = -np.sin(Alfa)
            Alfa_mat[2, 2] = -np.cos(Alfa)
            Alfa_mat[1, 1] = 1

    H_matrix = H @ Alfa_mat @ Alfa_mat.T

    # MCIL_memory_dealloc()