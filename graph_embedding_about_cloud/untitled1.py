# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:11:16 2019

@author: janae
"""

import random
import numpy as np
import tensorflow as tf
from collections import deque 

class S2V_NET:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		# init Q network
		self.create_S2V_QNetwork()

	def create_S2V_QNetwork(self):
		# network weights
		W_conv1 = self.weight_variable([8,8,4,32])
        
        