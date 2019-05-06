# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:54:55 2019

@author: janae
"""

import cv2
import sys
sys.path.append("game/")
import colud as game
from NewNet_Based_onDQN import BrainDQN
import numpy as np
ACTION_SIZE=100
# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playFlappyBird():
	# Step 1: init BrainDQN
	actions = ACTION_SIZE
	brain = BrainDQN(actions)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal,original_score = flappyBird.frame_step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	# Step 3.2: run the game
	#while 1!= 0:
	game_times = 60010
	max_score = 0
	average_score = 0
	count_score = 0
	episode_score = 0
	for episode in range(game_times):
		while True:
			action = brain.getAction()
			nextObservation,reward,terminal,episode_score = flappyBird.frame_step(action)
			nextObservation = preprocess(nextObservation)
			brain.setPerception(nextObservation,action,reward,terminal)
			if terminal:
				#print(episode_score)
				count_score += episode_score
				if episode_score>max_score:
					max_score = episode_score
					#print('最高分是：',max_score)
				if episode%1000==0 and episode>0:
					average_score = count_score/episode
					print('使用DQN算法，%d局游戏中，小鸟的平均得分是：%f,最高得分是%d'%(episode,average_score,max_score))
				break

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()
