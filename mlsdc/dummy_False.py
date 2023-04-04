#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:54:16 2023

@author: cwn4523
"""

from harmonicoscillator import minimax
from params import problem_params, collocation_params

iteration=minimax(problem_params, collocation_params)
iteration.coar_coeff()
iteration.coarQU_coeff()
iteration.coarse_to_fine()
iteration.save_data()
iteration.fine_coeff(dummy_data=False)
iteration.fineQ_coeff(dummy_data=False)
