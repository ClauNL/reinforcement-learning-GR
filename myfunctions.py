import math
import os
import random
import matplotlib

from pddlgym.core import InvalidAction, PDDLEnv
from pddlgym_planners.fd import FD; matplotlib.use('agg')

import imageio
import numpy as np





# Q-Learning

def qLearning(env, action_list, q = {}, 
                num_episodes = 500,
                max_steps = 100,

                alpha = 0.1,
                discount = 0.4,

                epsilon = 1,
                max_epsilon = 1,
                min_epsilon = 0.01,
                epsilon_decay_rate = 0.001):

    
    rewards = []

    for episode in range(num_episodes):
        # print(episode)
        obs, _ = env.reset()
        state = tuple(sorted(tuple(obs.literals)))
        rewards_current_episode = 0
        done = False

        for step in range(max_steps): 

            # explotación
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > epsilon:

                if state in q:
                    action = np.argmax(q[state])
                else:
                    selected_action = env.action_space.sample(obs)
                    action = action_list.index(selected_action)


            # exploración
            else:
                selected_action = env.action_space.sample(obs)
                action = action_list.index(selected_action)

            # ejecutar action            
            try:
                new_obs, reward, done, _ = env.step(action_list[action])
                new_state = tuple(sorted(tuple(new_obs.literals)))
            except InvalidAction:
                new_obs = obs
                new_state = state
                reward = 0
                done = False
                
            
            if state not in q:
                q[state] = [0.]*len(action_list)
            
            if new_state in q:
                max_next_state = np.max(q[new_state])
            else:
                max_next_state = 0.0

            # actualizar tabla Q
            q[state][action] = (1-alpha) * q[state][action] + alpha * (reward + discount * max_next_state )

        
            # actualizar estado
            obs = new_obs
            state = new_state

            rewards_current_episode += reward 

            if done == True: 
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate*episode)

        rewards.append(rewards_current_episode)

    print(len(q))

    rewards_per_100_episodes = np.split(np.array(rewards),num_episodes/100)
    count = 100

    print("********Average reward********\n")
    for r in rewards_per_100_episodes:
        print(count, ": ", str(sum(r/100)))
        count += 100


    return q, sum(rewards_per_100_episodes[-1]/100)
    


# Goal Recognition 

def goalRecognition(observation, policies, actions, real_goal, num_goals, distance):
    min_dist = math.inf
    predicted_goal = 0
    dist = 0
    distances = []

    for goal in range(num_goals):
        q = policies[goal]
        if distance == 'MaxUtil':
            dist = MaxUtil(q, actions, observation)
        elif distance == 'KL':
            dist = KL(q, actions, observation)
        elif distance == 'DP':
            dist = DP(q, actions, observation)
        

        # print(f'goal {goal}: {dist}' )
        distances.append(dist)

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    # print(f'Distance: {distance}.  Real Goal: {real_goal}  -  Predicted Goal: {predicted_goal} --> {predicted_goal == real_goal} ')
    
    return distances, predicted_goal == real_goal, predicted_goal







# Medidas de distancia

def MaxUtil(q, actions, observation):
    sum = 0
    for state, action in observation:
        if state in q:
            sum += q[state][actions.index(action)]

    return -sum

def policy(q_values):
    result = q_values
    total = sum(q_values)
    if total != 0:
        result = [value / total for value in q_values]

    return result

def KL(q, actions, observation):
    distances = []

    for state, action in observation:
        action_index = actions.index(action)
        
        if state not in q:
            pg = [1./len(actions) for _ in range(len(actions))]
    
        else:
            pg = policy(q[state])
        po = [0 for _ in range(len(actions))]
        po[action_index] = 1. 
        if pg[action_index] != 0:
            distances.append(po[action_index] * math.log2(po[action_index]/pg[action_index]))

    return np.mean(distances)


def DP(q, actions, observation):
    delta = 0.2

    for i, (state, action) in enumerate(observation):
        action_index = actions.index(action)
        p = 0
        if state in q:
            p = policy(q[state])[action_index]
        
        if p < delta:
            return -i 

    return -len(observation)





# Funciones para render

def create_video(env, q, action_list, name, path, verbose = False):

    images = []

    obs, _ = env.reset()
    state = tuple(sorted(tuple(obs.literals)))
    done = False

    start = env.render()
    end = env.render()

    for step in range(500):        
        images.append(env.render())
        
        if state in q:
            action = np.argmax(q[state]) 
        else:
            selected_action = env.action_space.sample(obs)
            action = action_list.index(selected_action)   
 
        if verbose:
            print(action_list[action])  
                
        try:
            obs, reward, done, _ = env.step(action_list[action])
            state = tuple(sorted(tuple(obs.literals)))
        except InvalidAction:
            done = False
    
        if done:
            images.append(env.render()) 
            end = env.render()                   
            break     


    if not os.path.exists(path): os.makedirs(path)
    imageio.imsave(f'{path}/start.png', start)
    imageio.imsave(f'{path}/end_{name}.png', end)
    imageio.mimwrite(f'{path}/video_{name}.mp4', images, fps=3)
    print('Wrote out video')




def create_video_from_obs(env, observation, path):

    images = []
    obs, _ = env.reset()
    done = False
    
    for state, action in observation:
        images.append(env.render())
        obs, reward, done, _ = env.step(action)

    images.append(env.render())
  

    if not os.path.exists(path): os.makedirs(path)
    imageio.mimwrite(f'{path}/video_OBS.mp4', images, fps=3)
    print('Wrote out video')
