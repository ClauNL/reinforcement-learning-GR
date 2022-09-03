#!/usr/bin/env python
import pickle
from re import template
import sys
import dill
from pddlgym.core import PDDLEnv

from myfunctions import *

BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

SKGRID = ['output/skgrid_gr/', 'output/skgrid_gr2/', 'output/skgrid_gr3/', 'output/skgrid_gr4/', 'output/skgrid_gr5/',
          'output/skgrid_gr6/', 'output/skgrid_gr7/', 'output/skgrid_gr8/', 'output/skgrid_gr9/', 'output/skgrid_gr10/']

SKGRID_TEST = ['output/skgrid_gr_test_show/']




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

        c = 0
        while last_reward < 0.9:
            # print(f'TRAINING TIME {c}')
            q, last_reward = qLearning(env, actions, q)
            c += 1


        policies.append(q)

        print(len(q))

        # with open(folder + 'qtables_data.txt', 'ab') as file:
        #     info = f'goal: {n} -  table size: {len(q)}'
        #     file.write(info)

        info = f'goal: {n} -  table size: {len(q)}\n'
        
        data.write(info)
        


    data.close()
    with open(folder + 'policies.pkl', 'wb') as file:
        dill.dump(policies, file)



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1: exit()
    folder = args[0]
    # gr_to_gym2(folder, output='OUT_PRUEBA', obs_per=100)
    train(folder)



# domain
# template
# hyps

#   # Setup directories, still needs some fixing here.
#     Path(output + '/' + d).mkdir(parents=True, exist_ok=True)
#     Path(output + '/' + d + '/problems').mkdir(parents=True, exist_ok=True)

#     # Complete domain and multiple problems
#     goals = []
#     for line in hypothesis:
#         goals.append(line)
#         # print(goals)
#     # complete_domain(domain, output + '/' + d, None)

#     # new_domain = open(output + '/' + d + "/domain.pddl", "w")
#     # new_domain.write(domain.read())
#     # new_domain.close()


#     for count, goal in enumerate(goals):
#         complete_problem2(template, goal, count, output + '/' + d)

    

# def complete_problem2(problem_file,  goal, number, path):
#     problem_file.seek(0)
#     print('Writing on: ' + path + "/problems/problem" + str(number) + ".pddl")
#     new_problem = open(path + "/problems/problem" + str(number) + ".pddl", "w")
#     counter = 1
#     init_state = False


#     for line in problem_file:
#         if '<HYPOTHESIS>' in line:
#             new_problem.write('\t' + goal)
#             continue
#         new_problem.write(line)

#     new_problem.close()


