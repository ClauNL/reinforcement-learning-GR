import math
from random import randint
import joblib
import numpy as np
from myfunctions import *
import pddlgym
import dill


# # domainPath = '/Users/cnegrin/Documents/TFG3/test/domain.pddl'
# # templatePath = '/Users/cnegrin/Documents/TFG3/test/template.pddl'
# # goalsPath = '/Users/cnegrin/Documents/TFG3/test/hyps.dat'


# # outPath = '/Users/cnegrin/Documents/TFG3/test'

# # with open(goalsPath) as file:
# #     goals = file.readlines()
# #     goals = [line.rstrip() for line in goals]


# # with open(templatePath, 'r') as file :
# #   filedata = file.read()


# # for i, goal in enumerate(goals):
# #     print(goal)
# #     filedata = filedata.replace('<HYPOTHESIS>', goal)

# #     with open(f'{outPath}/file{i}.pddl', 'w') as file:
# #         file.write(filedata)




# # def prueba(goalsPath):

# #     with open(goalsPath) as file:
# #         goals = file.readlines()
# #         goals = [line.rstrip() for line in goals]


# #     with open(templatePath, 'r') as file :
# #         filedata = file.read()


# #     for i, goal in enumerate(goals):
# #         print(goal)
# #         filedata = filedata.replace('<HYPOTHESIS>', goal)

# #         with open(f'{outPath}/file{i}.pddl', 'w') as file:
# #             file.write(filedata)


# # x = [0,1,2,3,4,5,6,7]
# # # print(len(x)/2)
# # # print(np.array_split(x, len(x)/2))
# # # print(x[:2])
# # # print(x[:4])
# # # print(x[:6])
# # # print(x[:8])

# # for i in range(0, len(x), 2):
# #     # print(i)
# #     # print(len(x))
# #     # if i == len(x)-2:
# #     #     print('ultimo')

# #     # print(i/2)
# #     # print(x[:i+2])

    
# #     print(x[i])
# #     print(x[i+1])



    

# # # print(len(x)/2 -1)


# # print(math.log(0.1))
# # print(math.log(0.1) * 0.1)

# # import dill


# folder = '/Users/cnegrin/Documents/paper_original/rl_goal_recognition/src/output/blocks_gr/'
# folder2 = '/Users/cnegrin/Documents/paper_original/rl_goal_recognition/src/output/blocks_gr/policies.pkl'
# folder3 = '/Users/cnegrin/Documents/paper_original/rl_goal_recognition/src/output/blocks_gr/actions.pkl'


# # with open(folder + 'policies.pkl', 'rb') as file:
# #     policies = dill.load(file)

# # # policies = dill.load('/Users/cnegrin/Documents/paper_original/rl_goal_recognition/src/output/blocks_gr/policies.pkl')


# # act = joblib.load(folder3)
# # print(act)

# # env_name = 'blocks'
# # env = pddlgym.make(f"PDDLEnvBlocks-v0") 
# # env.fix_problem_index(0)

# # init, _ = env.reset()

# # q = {'a': 1, 'b':2}

# # print(q)

# # for key, value in q.items():
# #     print(key, ' : ', value)



# # action_list = list(env.action_space.all_ground_literals(init, valid_only=False))
# # print(action_list)

# # actions = len(action_list)
# # print(actions)

# # state = tuple(sorted(tuple(init.literals)))
# # print(state)


# # q_table = {}

# # if state not in q_table:
# #     q_table[state] = [0.]*actions
# # action = np.argmax(q_table[state])  # index of max q action
# # print(action)

# # print(np.max(q_table[state]))  # max q value

# # print(q_table)
# # print(action_list[action])

# # # obs, reward, done, _ = env.step(action_list[action])
# # # next_state = tuple(sorted(tuple(obs.literals)))

# # # action = self.agent_step(reward, next_state)

# # print(randint(0, actions-1))

# # print(q_table[state][5])

# import dill 
# # with open('/Users/cnegrin/Documents/TFG-NEW/output/blocks_gr_test/actions.pkl', 'rb') as file:
# #     actions = dill.load(file)

# # print(actions)

# # selected_act = actions[1]

# # print(actions.index(selected_act))



# with open('/Users/cnegrin/Documents/TFG-NEW/output/blocks_gr_test/policies.pkl', 'rb') as file:
#     policies = dill.load(file)

# q = policies[0]
# for state, action in q.items():
#     print(type(q[state]))


# scores = [-4, 0, -2, 0]
# rankings = sorted(((goal, div) for (goal, div) in enumerate(scores)), key=lambda tup: tup[1])
# print(rankings)

# correct True
# goal 0
# ranking  [(0, -4), (2, -2), (1, 0), (3, 0)]


# def measure_confusion(r_output):
#     prediction = r_output[0]   
#     ranking = r_output[-1]

#     head = ranking[0]    (0, -4)
#     tail = ranking[1:]    (2, -2), (1, 0), (3, 0)
#     fn = int(not prediction)    True --> 0   False --> 1
#     fp = 0
#     tn = 0
#     if prediction:       
#         for goal_value in tail:
#             if goal_value[1] == head[1]:
#                 fp += 1
#             else:
#                 tn += 1    
#     else:
#         fp = 1
#         for goal_value in tail[:-1]:
#             if goal_value[1] == head[1]:
#                 fp += 1
#             else:
#                 tn += 1

#     #      tp               fn                   fp  tn       
#     return int(prediction), fn, fp, tn


obs_traces = []

folder = 'output/skgrid_gr_test_show/'
d = 'output/skgrid_gr_test_show/'
output = 'output/skgrid_gr_test_show/obs_show/'
# create_observabilities(d, output, ind=0)

with open(folder + 'policies.pkl', 'rb') as file:
    policies = dill.load(file)
with open(folder + 'actions.pkl', 'rb') as file:
    actions = dill.load(file)

with open(folder + 'obs1.0.pkl', 'rb') as file:
    obs_traces.append(pickle.load(file))

# print(actions)

# # [move(dir-down:direction), move(dir-left:direction), move(dir-up:direction), move(dir-right:direction)]

# up = actions[2]
# right = actions[3]

# ruta = [up, up, up, up, up, up, right, right, right]
# create_observabilities2(d, output, ruta, ind=0)

pol = policies[0]
trajectory = obs_traces[0]
epsilon = 0.
# converts a trajectory from a planner to a policy
# where the taken action has 99.99999% probability
trajectory_as_policy = {}
for state, action in trajectory:
    action_index = actions.index(action)
    actions_len = len(actions)
    qs = [1e-6 + epsilon/actions_len for _ in range(actions_len)]
    qs[action_index] = 1. - 1e-6 * (actions_len-1) - epsilon
    trajectory_as_policy[tuple(state)] = qs

print(trajectory_as_policy.values())

softmax_policy = {state: policy2(pol[state]) for state in pol.keys()}

print(softmax_policy.values())