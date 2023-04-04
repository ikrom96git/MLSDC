#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:09:44 2023

@author: cwn4523
"""

class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)
