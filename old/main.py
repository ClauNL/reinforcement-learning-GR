from collections import defaultdict
import glob
import os
from tracemalloc import DomainFilter
from idna import valid_contextj
import joblib
import matplotlib
import numpy as np; matplotlib.use('agg')

import pddlgym
import imageio

from myfunctions import *



def init_enviroment(env_name, problem_dir):
    env = pddlgym.make(f"PDDLEnv{env_name}-v0", raise_error_on_invalid_action=False, dynamic_action_space = True, problem_dir = problem_dir)    
    return env

def get_qtable(results_dir, goal):
    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl')
    return q



def learn_all(env, results_dir):
    # crear enviroment
    num_problems = len(env.problems)
    for goal in range(num_problems):
        print(f'goal: {goal}')

        env.fix_problem_index(goal)

        q_name = f'q_g{goal}'
        q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
        q = qLearning(env, q, num_episodes = 500, max_steps = 1000)  # Q Learning
        writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q
        video(env, q, q_name, results_dir)    # crear video con la solucion


def learn_goal(env, results_dir, goal):

    print(f'goal: {goal}')
    env.fix_problem_index(goal)
    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    q = qLearning(env, q, num_episodes = 100, max_steps = 1000)  # Q Learning
    writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q
    video(env, q, q_name, results_dir)    # crear video con la solucion




def get_observation(env, results_dir, goal):
    env.fix_problem_index(goal)
    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    obs = getObservation(env, q, 200)  # Q Learning
    return obs



def goalRecognition(env, observation, distance, results_dir):
    min_dist = math.inf
    predicted_goal = 0
    dist = 0

    for goal in range(len(env.problems)):

        q_name = f'q_g{goal}'
        q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q 
          
        if distance == 'MaxUtil':
            dist = MaxUtil(q, observation)
        elif distance == 'KL':
            dist = KL(q, observation)
        
        elif distance == 'DP':
            dist = DP(q, observation)
        

        print(f'goal {goal}: {dist}' )

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    print(f'predicted goal: {predicted_goal}')
    return predicted_goal



domain = 'blocks'
problem = 'problem10'


problem_dir = f'/Users/cnegrin/Documents/TFG3/{domain}_test/{problem}'
results_dir = f'/Users/cnegrin/Documents/TFG3/results_prueba2/{domain}/{problem}'
if not os.path.exists(results_dir): os.makedirs(results_dir)



# env = init_enviroment(domain.capitalize(), problem_dir)
env = init_enviroment('Blocks_test', problem_dir)


# LEARN 
# learn_all(env, results_dir)
learn_goal(env, results_dir, 0)


# INFER

# goal = 1
# print(f'selected goal: {goal}')

# obs = get_observation(env, results_dir, goal)
# goalRecognition(env, obs, 'MaxUtil',  results_dir)



