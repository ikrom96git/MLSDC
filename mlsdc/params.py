#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:53:17 2023

@author: cwn4523
"""

problem_params=dict()
problem_params['kappa']=10
problem_params['mu']=3
problem_params['u0']=[4.0, 6]
problem_params['dt']=0.5
problem_params['Tend']=2.0

collocation_params=dict()
collocation_params['quad_type']='GAUSS'
collocation_params['num_nodes']=[5,3]
