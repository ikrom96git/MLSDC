#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:48:11 2023

@author: cwn4523
"""
import glob
import numpy as np
from harmonicoscillator import MLSDC, Plot_residual
from params import problem_params, collocation_params
import matplotlib.pyplot as plt
def new_coarse():
    name=glob.glob('./data/*.npy')
    Res_pos = []
    Res_vel = []

    for ii in range(len(name)):
        if 'res' in name[ii]:

            res=np.load(name[ii])
            res_pos, res_vel=np.split(res, 2)
            Res_pos=np.append(np.max(res_pos), Res_pos)
            Res_vel=np.append(np.max(res_vel), Res_vel)
    return Res_pos, Res_vel

def SDC():
    iteration=MLSDC(problem_params, collocation_params)
    U_MLSDC, R_MLSDC=iteration.MLSDC_iter(Kiter=4)
    U_SDC, R_SDC=iteration.SDC_iter(Kiter=4)
    time=iteration.fine.coll.coll.nodes
    mlsdc_pos, mlsdc_vev=np.split(U_MLSDC, 2)
    sdc_pos, sdc_vel=np.split(U_SDC, 2)
    # plt.plot(time, mlsdc_pos, label='MLSDC')
    # plt.plot(time, sdc_pos, label='SDC')
    # plt.xlabel('nodes points')
    # plt.ylabel('Solution')
    # plt.title('Position on the single time step')
    # plt.legend()
    # plt.show()
    return R_MLSDC, R_SDC

if __name__=='__main__':
    res_pos, res_vel = new_coarse()
    res_pos=np.flip(res_pos)
    R_MLSDC, R_SDC=SDC()
    title=['MLSDC(CH)', 'MLSDC', 'SDC']
    residual=np.block([ [res_pos], [R_MLSDC[:,0]], [R_SDC[:,0]]])
    Plot_residual(np.arange(1, 5), residual, titles=title)
