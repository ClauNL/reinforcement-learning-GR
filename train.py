import pickle
import dill
from pddlgym.core import PDDLEnv

from myfunctions import *

BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

BLOCKS2 = [ 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

BLOCKS3 = [ 'output/blocks_gr2/', 'output/blocks_gr9/', 'output/blocks_gr10/']



BLOCKS_TEST = ['output/blocks_gr_test/']

BLOCKS_TEST2 = ['output/blocks_gr2/']


HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

HANOI2 = ['output/hanoi_gr10/']
HANOI3 = ['output/hanoi_gr4/', 'output/hanoi_gr5/', 'output/hanoi_gr6/']
HANOI4 = ['output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']



SKGRID = ['output/skgrid_gr/', 'output/skgrid_gr2/', 'output/skgrid_gr3/', 'output/skgrid_gr4/', 'output/skgrid_gr5/',
          'output/skgrid_gr6/', 'output/skgrid_gr7/', 'output/skgrid_gr8/', 'output/skgrid_gr9/', 'output/skgrid_gr10/']

SKGRID_TEST = ['output/skgrid_gr_test_show/']



# for folder in BLOCKS:
        
#     print('Training domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#     # obs_traces = []
#     n_goals = len(env.problems)
#     # real_goal = 0

  


#     # get actions
#     env.fix_problem_index(0)
#     init, _ = env.reset()
#     actions = list(env.action_space.all_ground_literals(init, valid_only=False))
#     with open(folder + 'actions.pkl', 'wb') as file:
#         dill.dump(actions, file)

    
#     # with open(folder + 'actions.pkl', 'rb') as file:
#     #     actions = dill.load(file)

    



#     policies = []
#     for n in range(n_goals):
#         print('Training problem:', n)

#         env.fix_problem_index(n)
#         init, _ = env.reset()
    
#         # build method to learn policy

#         last_reward = 0
#         q = {}
#         # with open(folder + 'policies.pkl', 'rb') as file:
#         #     policies = dill.load(file)
#         #     q = policies[n]
    
#         c = 0
#         while last_reward < 0.98:
#             print(f'TRAINING TIME {c}')
#             q, last_reward = qLearning2(env, actions, q)
#             c += 1


#         policies.append(q)

#         # for key, value in q.items():
#         #     print(key, ' : ', value)

#         # print(len(q))


#         # for key, value in q.items():
#         #     print(value)

#         print(len(q))

#         # print(actions)



#     with open(folder + 'policies.pkl', 'wb') as file:
#         dill.dump(policies, file)




for folder in SKGRID_TEST:
        
    print('Training domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
    env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
    # obs_traces = []
    n_goals = len(env.problems)
    # real_goal = 0

  


    # get actions
    env.fix_problem_index(0)
    init, _ = env.reset()
    # actions = list(env.action_space.all_ground_literals(init, valid_only=False))
    # with open(folder + 'actions.pkl', 'wb') as file:
    #     dill.dump(actions, file)

    
    with open(folder + 'actions.pkl', 'rb') as file:
        actions = dill.load(file)

    



    policies = []


    # with open(folder + 'policies.pkl', 'rb') as file:
    #     policies = dill.load(file)
    #     q = policies[n]

    for n in range(n_goals):
        print('Training problem:', n)

        env.fix_problem_index(n)
        init, _ = env.reset()
    
        # build method to learn policy

        last_reward = 0
        q = {}
        # with open(folder + 'policies.pkl', 'rb') as file:
        #     policies = dill.load(file)
        #     q = policies[n]
    
        c = 0
        while last_reward < 0.98:
            print(f'TRAINING TIME {c}')
            q, last_reward = qLearning2(env, actions, q)
            c += 1


        policies.append(q)

        # for key, value in q.items():
        #     print(key, ' : ', value)

        # print(len(q))


        # for key, value in q.items():
        #     print(value)

        print(len(q))

        # print(actions)



    with open(folder + 'policies.pkl', 'wb') as file:
        dill.dump(policies, file)










