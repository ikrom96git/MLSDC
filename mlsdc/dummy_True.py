#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:57:18 2023

@author: cwn4523
"""

from harmonicoscillator import minimax
from params import problem_params, collocation_params

iteration=minimax(problem_params, collocation_params)

iteration.fine_coeff()
