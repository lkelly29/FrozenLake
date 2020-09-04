# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:15:36 2020

@author: Kelly


"""
import gym
import numpy as np

env = gym.make('FrozenLake-v0', is_slippery = False)

# every state has a row and every action have a value
q_table = np.zeros([env.observation_space.n, env.action_space.n])  

env.render()
wins = 0 # count number of wins to goal

for i_episode in range(1, 10000):
    state = env.reset()
    
    for x in range(1,150):
        currentstate = state
        if np.random.uniform(0,1) < 0.2:
            action = env.action_space.sample() # takes action randomly
        else:
            action = np.argmax(q_table[state]) # takes action with largest reward
        
        state, reward, done, info = env.step(action)
        
        if done and reward != 1:
             q_table[currentstate, action] = -1
        else:
             q_table[currentstate, action] = q_table[state][np.argmax(q_table[state])]
        
        # step on frozen ground (good)
        if done and reward == 1:
            wins += 1
            q_table [currentstate, action] = reward+100
        
        if done:
            break

print(q_table)
print(wins)
env.close()

    
    