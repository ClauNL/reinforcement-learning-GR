import pickle
import dill
import pandas as pd
from pddlgym.core import PDDLEnv
import joblib

from myfunctions import *

BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

BLOCKS_TEST = ['output/blocks_gr_test/']

BLOCKS_TEST2 = ['output/blocks_gr/']


HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

HANOI2 = ['output/hanoi_gr/']

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
            dist = MaxUtil2(q, actions, observation)
        elif distance == 'KL':
            dist = KL2(q, actions, observation)
        elif distance == 'KL2':
            dist = kl_divergence_norm_softmax(q, actions, observation)
        elif distance == 'DP':
            dist = DP2(q, actions, observation)
        

        print(f'goal {goal}: {dist}' )
        distances.append(dist)

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    # print(f'predicted goal: {predicted_goal}')

    print(f'Distance: {distance}.  Real Goal: {real_goal}  -  Predicted Goal: {predicted_goal} --> {predicted_goal == real_goal} ')
    
    correct = 0
    if predicted_goal == real_goal:
        correct = 1
    return distances, predicted_goal == real_goal, predicted_goal



# for domain in [BLOCKS, HANOI, SKGRID]:
# # for domain in [HANOI]:


#     results = pd.DataFrame()
#     for folder in domain:

#         print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#         env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#         obs_traces = []
#         obs = []
#         n_goals = len(env.problems)
#         real_goal = 0
#         with open(folder + 'real_hypn.dat', 'rb') as goal:
#             real_goal = int(goal.readline())

    

#         with open(folder + 'policies.pkl', 'rb') as file:
#             policies = dill.load(file)
#         with open(folder + 'actions.pkl', 'rb') as file:
#             actions = dill.load(file)
            

#         with open(folder + 'obs1.0.pkl', "rb") as input_file:
#             obs_traces.append(pickle.load(input_file))


        

#         for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
#             distances, correct, pred_goal = goalRecognition(obs_traces[0], policies, actions, real_goal, n_goals, metric)


#             x = {'problem': folder,
#                 'metric': metric, 'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
#                 'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
#             results = results.append(x, ignore_index = True)

        
#     print(results)
#     if domain == BLOCKS: name = 'blocks'
#     if domain == HANOI: name = 'hanoi'
#     if domain == SKGRID: name = 'skgrid'
#     results.to_csv(f'results/{name}.csv', index=False)






# for domain in [BLOCKS, HANOI, SKGRID]:
for domain in [HANOI]:


    results = pd.DataFrame()
    for folder in domain:

        print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
        env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        obs_traces = []
        obs = []
        n_goals = len(env.problems)
        real_goal = 0
        with open(folder + 'real_hypn.dat', 'rb') as goal:
            real_goal = int(goal.readline())

    
        with open(folder + 'policies.pkl', 'rb') as file:
            policies = dill.load(file)
        with open(folder + 'actions.pkl', 'rb') as file:
            actions = dill.load(file)

        for o in [0.1, 0.3, 0.5, 0.7, 1.0]:
        # for o in [0.1]:
            with open(folder + 'obs' + str(o) +'.pkl', "rb") as input_file:
                obs_traces.append(pickle.load(input_file))

        test = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 1.0}


        for i, trace in enumerate(obs_traces):

            # for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
            for metric in ['MaxUtil', 'DP', 'KL']:
            # for metric in [ 'KL2']:
                distances, correct, pred_goal = goalRecognition(trace, policies, actions, real_goal, n_goals, metric)


                x = {'problem': folder, 'obs': test[i],
                    'metric': metric, 
                    'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
                    'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
                x_dictionary = pd.DataFrame([x])
                # results = results.append(x, ignore_index = True)
                
                results = pd.concat([results, x_dictionary], ignore_index=True)

            


        
    print(results)
    if domain == BLOCKS: name = 'blocks'
    if domain == HANOI: name = 'hanoi'
    if domain == SKGRID: name = 'skgrid'
    if domain == SKGRID_TEST: name = 'skgrid_test'
    results.to_csv(f'results_PROBANDO/{name}.csv', index=False)















# # for domain in [BLOCKS, HANOI, SKGRID]:
# for domain in [SKGRID_TEST]:


#     results = pd.DataFrame()
#     results2 = pd.DataFrame()
#     results3 = {}
#     for folder in domain:

#         print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#         env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#         obs_traces = []
#         obs = []
#         n_goals = len(env.problems)
#         real_goal = 0
#         with open(folder + 'real_hypn.dat', 'rb') as goal:
#             real_goal = int(goal.readline())

    

#         with open(folder + 'policies.pkl', 'rb') as file:
#             policies = dill.load(file)
#         with open(folder + 'actions.pkl', 'rb') as file:
#             actions = dill.load(file)

#         for o in [0.1, 0.3, 0.5, 0.7, 1.0]:
#         # for o in [0.1]:
#             with open(folder + 'obs' + str(o) +'.pkl', "rb") as input_file:
#                 obs_traces.append(pickle.load(input_file))

#         test = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 1.0}


#         for i, trace in enumerate(obs_traces):

#             for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
#             # for metric in [ 'KL']:
#                 distances, correct, pred_goal = goalRecognition(trace, policies, actions, real_goal, n_goals, metric)


#                 x = {'problem': folder, 'obs': test[i],
#                     'metric': metric, 'scores': distances, 
#                     # 'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
#                     'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct, 'correct_num': int(correct)}
#                 results = results.append(x, ignore_index = True)

#                 # if correct == 0: correct2 = False
#                 # if correct == 1: correct2 = True
#                 x2 = {'problem': folder, 'obs': test[i],
#                     'metric': metric, 'scores': sorted(((goal, div) for (goal, div) in enumerate(distances)), key=lambda tup: tup[1]),
#                     'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
#                 results2 = results2.append(x2, ignore_index = True)

#                 n = f'{folder}_{i}_{metric}'
#                 results3[n] = x2



        
#     print(results)
#     if domain == BLOCKS: name = 'blocks'
#     if domain == HANOI: name = 'hanoi'
#     if domain == SKGRID: name = 'skgrid'
#     if domain == SKGRID_TEST: name = 'skgrid_test'
#     results.to_csv(f'results/{name}.csv', index=False)
#     results2.to_csv(f'results2/{name}.csv', index=False)


#     with open(f'results/{name}.pkl', 'wb') as file:
#         dill.dump(results, file)
#     with open(f'results2/{name}.pkl', 'wb') as file:
#         dill.dump(results2, file)
#     with open(f'results3/{name}.pkl', 'wb') as file:
#         dill.dump(results3, file)


#     print(results2)
#     print(results2.dtypes)




# results = pd.DataFrame()

# for folder in HANOI:

#     print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     obs = []
#     n_goals = len(env.problems)
#     real_goal = 0
#     with open(folder + 'real_hypn.dat', 'rb') as goal:
#         real_goal = int(goal.readline())

  

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))


    

#     for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
#         distances, correct, pred_goal = goalRecognition(obs_traces[0], policies, actions, real_goal, n_goals, metric)


#         x = {'problem': folder,
#             'metric': metric, 'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
#             'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
#         results = results.append(x, ignore_index = True)

    
# print(results)
# results.to_csv('results/hanoi.csv', index=False)




# results = pd.DataFrame()

# for folder in BLOCKS:

#     print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     obs = []
#     n_goals = len(env.problems)
#     real_goal = 0
#     with open(folder + 'real_hypn.dat', 'rb') as goal:
#         real_goal = int(goal.readline())

  

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))


    

#     for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
#         distances, correct, pred_goal = goalRecognition(obs_traces[0], policies, actions, real_goal, n_goals, metric)


#         x = {'problem': folder,
#             'metric': metric, 'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
#             'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
#         results = results.append(x, ignore_index = True)

    
# print(results)
# results.to_csv('results/blocks.csv', index=False)








# results = pd.DataFrame()

# for folder in SKGRID:

#     print('Recognize domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
#     env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
#     obs_traces = []
#     obs = []
#     n_goals = len(env.problems)
#     real_goal = 0
#     with open(folder + 'real_hypn.dat', 'rb') as goal:
#         real_goal = int(goal.readline())

  

#     with open(folder + 'policies.pkl', 'rb') as file:
#         policies = dill.load(file)
#     with open(folder + 'actions.pkl', 'rb') as file:
#         actions = dill.load(file)

#     with open(folder + 'obs1.0.pkl', "rb") as input_file:
#         obs_traces.append(pickle.load(input_file))


    

#     for metric in ['MaxUtil', 'DP', 'KL', 'KL2']:
#         distances, correct, pred_goal = goalRecognition(obs_traces[0], policies, actions, real_goal, n_goals, metric)


#         x = {'problem': folder,
#             'metric': metric, 'g0': distances[0], 'g1': distances[1], 'g2': distances[2], 'g3': distances[3], 
#             'real_goal': real_goal, 'pred_goal': pred_goal, 'correct': correct}
#         results = results.append(x, ignore_index = True)

    
# print(results)
# results.to_csv('results/skgrid.csv', index=False)



    
    # for n in range(n_goals):
    #     print(f'GOAL {n}')

    #     value = MaxUtil2(policies[n], actions, obs_traces[0])
    #     print(value)



    

    # sum = 0
    # for i in range(0, len(observation), 2):
    #     state = observation[i]
    #     action = observation[i+1]
    #     if state in q:
    #         sum += q[state].get(action, 0.0)


    
        



    #     accumulated_q = 0
    # for state, action in trajectory:
    #     accumulated_q += p1.get_all_q_values(state)[actions.index(action)]








    # for n in range(n_goals):
    #     env.fix_problem_index(n)
    #     create_video2(env, policies[n], actions, n, folder + 'results/', verbose = True)


    



    # policies = []
    # for n in range(n_goals):
    #     print('Training problem:', n)

    #     env.fix_problem_index(n)
    #     init, _ = env.reset()
    
    #     # build method to learn policy

    #     last_reward = 0
    #     q = {}
    #     c = 0
    #     while last_reward < 0.98:
    #         print(f'TRAINING TIME {c}')
    #         q, last_reward = qLearning2(env, actions, q)
    #         c += 1


    #     policies.append(q)

    #     # for key, value in q.items():
    #     #     print(key, ' : ', value)

    #     # print(len(q))


    #     # for key, value in q.items():
    #     #     print(value)

    #     print(len(q))

    #     # print(actions)



    # with open(folder + 'policies.pkl', 'wb') as file:
    #     dill.dump(policies, file)













