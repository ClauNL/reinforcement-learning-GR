#!/usr/bin/env python
import pickle
import sys
import dill
import pandas as pd
from pddlgym.core import PDDLEnv

from myfunctions import *



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

    with open(folder + 'obs' + str(o) +'.pkl', "rb") as input_file:
        obs_traces.append(pickle.load(input_file))



    for i, trace in enumerate(obs_traces):

        distances, correct, pred_goal = goalRecognition(trace, policies, actions, real_goal, n_goals, metric)

        x = {'problem': folder, 
            'obs': o,
            'metric': metric, 
            'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
            'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
        x_dictionary = pd.DataFrame([x])
        
        results = pd.concat([results, x_dictionary], ignore_index=True)

        
        
    # print(results)
    Path(f'{folder}/results_gr/').mkdir(parents=True, exist_ok=True)
    results.to_csv(f'{folder}/results_gr/{metric}_{o}.csv', index=False)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3: exit()
    folder = args[0] + '/'
    metric = args[1]
    obs = args[2]
    recognize(folder, metric, obs)












