#!/usr/bin/env python
import sys
import dill
from pddlgym.core import PDDLEnv

from myfunctions import *



def train(folder):
    print('Training domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
    env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
    n_goals = len(env.problems)

    # get actions
    env.fix_problem_index(0)
    init, _ = env.reset()
    actions = list(env.action_space.all_ground_literals(init, valid_only=False))
    with open(folder + 'actions.pkl', 'wb') as file:
        dill.dump(actions, file)


    policies = []
    data = open(folder + 'qtables_data.txt', "w")
    for n in range(n_goals):
        print('Training problem:', n)

        env.fix_problem_index(n)
        init, _ = env.reset()
    

        last_reward = 0
        q = {}

        while last_reward < 0.9:
            q, last_reward = qLearning(env, actions, q)


        policies.append(q)

        info = f'goal: {n} -  table size: {len(q)}\n'
        
        data.write(info)
        


    data.close()
    with open(folder + 'policies.pkl', 'wb') as file:
        dill.dump(policies, file)



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1: exit()
    folder = args[0] + '/'
    train(folder)



