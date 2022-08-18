import pickle
import dill
from pddlgym.core import PDDLEnv

from myfunctions import *
# from pddlgym.rendering.sokoban import render as sokoban_render
from render.sokoban import render as sokoban_render
from render.blocks import render as blocks_render
from render.hanoi import render as hanoi_render




BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

BLOCKS_TEST = ['output/blocks_gr_test/']
BLOCKS_TEST2 = ['output/blocks_gr2/']


HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

HANOI2 = ['output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

SKGRID = ['output/skgrid_gr/', 'output/skgrid_gr2/', 'output/skgrid_gr3/', 'output/skgrid_gr4/', 'output/skgrid_gr5/',
          'output/skgrid_gr6/', 'output/skgrid_gr7/', 'output/skgrid_gr8/', 'output/skgrid_gr9/', 'output/skgrid_gr10/']

SKGRID_TEST = ['output/skgrid_gr_test_show/']



for folder in SKGRID_TEST:

    print('Showing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
    env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', sokoban_render, raise_error_on_invalid_action=True, dynamic_action_space=True)
    obs_traces = []
    n_goals = len(env.problems)
    real_goal = 0

    with open(folder + 'policies.pkl', 'rb') as file:
        policies = dill.load(file)
    with open(folder + 'actions.pkl', 'rb') as file:
        actions = dill.load(file)

    with open(folder + 'obs1.0.pkl', "rb") as input_file:
        obs_traces.append(pickle.load(input_file))

    # for n in range(n_goals):
    #     env.fix_problem_index(n)
    #     create_video2(env, policies[n], actions, n, folder + 'results/', verbose = True)
    
    
    env.fix_problem_index(0)
    create_video_from_obs(env, obs_traces[0], folder + 'results/')


# for folder in SKGRID:

#     print('Showing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', sokoban_render, raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     n_goals = len(env.problems)
#     real_goal = 0

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))

#     for n in range(n_goals):
#         env.fix_problem_index(n)
#         create_video2(env, policies[n], actions, n, folder + 'results/', verbose = True)

#     env.fix_problem_index(0)
#     create_video_from_obs(env, obs_traces[0], folder + 'results/')



# for folder in HANOI:
    
#     print('Showing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', hanoi_render, raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     n_goals = len(env.problems)
#     real_goal = 0

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))

#     for n in range(n_goals):
#         env.fix_problem_index(n)
#         create_video2(env, policies[n], actions, n, folder + 'results/', verbose = True)

#     env.fix_problem_index(0)
#     create_video_from_obs(env, obs_traces[0], folder + 'results/')



    
# for folder in BLOCKS:
    
#     print('Showing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', blocks_render, raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     n_goals = len(env.problems)
#     real_goal = 0

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))

#     for n in range(n_goals):
#         env.fix_problem_index(n)
#         create_video2(env, policies[n], actions, n, folder + 'results/', verbose = True)

#     env.fix_problem_index(0)
#     create_video_from_obs(env, obs_traces[0], folder + 'results/')

