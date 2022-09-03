#!/usr/bin/env python
import pickle
import sys
import dill
import pandas as pd
from pddlgym.core import PDDLEnv
import joblib

from myfunctions import *

BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

SKGRID = ['output/skgrid_gr/', 'output/skgrid_gr2/', 'output/skgrid_gr3/', 'output/skgrid_gr4/', 'output/skgrid_gr5/',
          'output/skgrid_gr6/', 'output/skgrid_gr7/', 'output/skgrid_gr8/', 'output/skgrid_gr9/', 'output/skgrid_gr10/']

SKGRID_TEST = ['output/skgrid_gr_test_show/']



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
        
        print(f'goal {goal}: {dist}' )
        distances.append(dist)

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    print(f'Distance: {distance}.  Real Goal: {real_goal}  -  Predicted Goal: {predicted_goal} --> {predicted_goal == real_goal} ')
    
    return distances, predicted_goal == real_goal, predicted_goal





# ESTTEEEEEEEEEE ESE EL QUE FUNCIONAAAAAAAAAAAAAAAAAAAA
# for domain in [BLOCKS, HANOI, SKGRID]:
# for domain in [SKGRID_TEST]:



def recognize(folder, metric, o):
    results = pd.DataFrame()

    print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
    env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
    obs_traces = []
    n_goals = len(env.problems)
    real_goal = 0
    with open(folder + 'real_hypn.dat', 'rb') as goal:
        real_goal = int(goal.readline())

    with open(folder + 'policies.pkl', 'rb') as file:
        policies = dill.load(file)
    with open(folder + 'actions.pkl', 'rb') as file:
        actions = dill.load(file)

    # for o in [0.1, 0.3, 0.5, 0.7, 1.0]:
    with open(folder + 'obs' + str(o) +'.pkl', "rb") as input_file:
        obs_traces.append(pickle.load(input_file))

    # obs_number = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 1.0}


    for i, trace in enumerate(obs_traces):

        # for metric in ['MaxUtil', 'DP', 'KL']:
        distances, correct, pred_goal = goalRecognition(trace, policies, actions, real_goal, n_goals, metric)


        x = {'problem': folder, 
            # 'obs': obs_number[i],
            'obs': o,
            'metric': metric, 
            'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
            'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
        x_dictionary = pd.DataFrame([x])
        
        results = pd.concat([results, x_dictionary], ignore_index=True)

        
        
    print(results)
    Path(f'{folder}/results_gr/').mkdir(parents=True, exist_ok=True)
    results.to_csv(f'{folder}/results_gr/{metric}_{o}.csv', index=False)
    return results
    # if domain == BLOCKS: name = 'blocks'
    # if domain == HANOI: name = 'hanoi'
    # if domain == SKGRID: name = 'skgrid'
    # if domain == SKGRID_TEST: name = 'skgrid_test'
    # results.to_csv(f'results_PROBANDO/{name}.csv', index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3: exit()
    folder = args[0]
    metric = args[1]
    obs = args[2]
    # gr_to_gym2(folder, output='OUT_PRUEBA', obs_per=100)
    recognize(folder, metric, obs)












