#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:53:17 2023

@author: cwn4523
"""

problem_params=dict()
problem_params['kappa']=1
problem_params['mu']=0
problem_params['u0']=[1.0, 0]
problem_params['dt']=0.1
problem_params['Tend']=2.0

collocation_params=dict()
collocation_params['quad_type']='LOBATTO'
collocation_params['num_nodes']=[5,3]
