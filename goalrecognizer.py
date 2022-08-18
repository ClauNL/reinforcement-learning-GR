import glob
import os
import matplotlib
from matplotlib.pyplot import jet
from pddlgym.rendering import blocks, hanoi; matplotlib.use('agg')
import pddlgym
import joblib
import sys, getopt
import pandas as pd

from pathlib import Path  
file = Path(__file__).resolve()  
directory = file.parents[0]  

from myfunctions import *



  
def get_observation(env, results_dir, goal):
    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    obs = getObservation(env, q, 200)  # Q Learning
    return obs



def goalRecognition(observation, distance, results_dir, num_goals):
    min_dist = math.inf
    predicted_goal = 0
    dist = 0

    for goal in range(num_goals):

        q_name = f'q_g{goal}'
        q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q 
          
        if distance == 'MaxUtil':
            dist = MaxUtil(q, observation)
        elif distance == 'KL':
            dist = KL(q, observation)
        elif distance == 'DP':
            dist = DP(q, observation)
        

        print(f'goal {goal}: {dist}\n' )

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    # print(f'predicted goal: {predicted_goal}')
    return predicted_goal




   

# domain  dir  
# domain  dir  metric

def main():
    args = sys.argv[1:]

    if len(args) != 2 | len(args) != 3: exit()

    metrics = ['MaxUtil','KL','DP']

    if args[0] == 'nav': env_name = 'nav'
    if args[0] == 'blocks': env_name = 'blocks_operator_actions'
    if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
    
    dir = args[1]
    problem_dir = f'{dir}/problem_files'
    results_dir = f'{dir}/RESULTS'

    if len(args) == 3:
        metrics = [args[2]]


        

    num_goals = len([f for f in glob.glob(os.path.join(results_dir, "*.pkl"))])
    num_files = len([f for f in glob.glob(os.path.join(problem_dir, "*.pddl"))])
    num_starts = int(num_files/num_goals)
    print(f'num goals: {num_goals}')
    print(f'num files: {num_files}')
    print(f'num starts: {num_starts}')
    

 
    results = pd.DataFrame()  # start goal pred_goal metric

    results2 = pd.DataFrame()  # start goal pred_goal metric  step

    # sum = 0
    # for metric in ['MaxUtil','KL','DP']:
    for metric in metrics:
        for j in range(num_starts):
            for i in range(num_goals):


                print(f'start: {j}')
                print(f'goal: {i}')


                env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
                env.fix_problem_index(num_goals*j + i)
                

                q_name = f'q_g{i}'
                q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
                obs = getObservation(env, q, 200)  # Q Learning
                # for o in range(0, len(obs), 2):
                #     print(obs[o+1])

                for o in range(0, len(obs), 2):
                    # print(obs[:o])
                    predicted_goal = goalRecognition(obs[:o+2], metric,  results_dir, num_goals)
                    x = {'start': j, 'goal': i, 'pred_goal': predicted_goal, 'metric': metric, 'step': o/2 }
                    results2 = results2.append(x, ignore_index = True)

                    if o == len(obs)-2:
                        x = {'start': j, 'goal': i, 'pred_goal': predicted_goal, 'metric': metric }
                        results = results.append(x, ignore_index = True)
                        print(f'predicted goal: {predicted_goal}\n')




                # predicted_goal = goalRecognition(obs, metric,  results_dir, num_goals)

                # print(f'predicted goal: {predicted_goal}\n')

                # x = {'start': j, 'goal': i, 'pred_goal': predicted_goal, 'metric': metric }
                # results = results.append(x, ignore_index = True)

                # x = {'start': j, 'goal': i, 'pred_goal': predicted_goal, 'metric': metric, 'step': math.ceil(len(obs)/2) }
                # results2 = results2.append(x, ignore_index = True)

                


                

    print(results)
    results.to_csv(f'{dir}/results.csv', index=False)

    print(results2)
    results2.to_csv(f'{dir}/results2.csv', index=False)







main()



# python3 goalrecognizer.py 'nav' test_files/nav/problem2/problem_files test_files/nav/problem2/RESULTS


# python3 goalrecognizer.py 'nav' test_files/nav/problem3









# def learn(env, results_dir, goal):

#     q_name = f'q_g{goal}'
#     q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
#     q = qLearning(env, q, num_episodes = 200, max_steps = 10000)  # Q Learning
#     writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q
#     video(env, q, q_name, results_dir)    # crear video con la solucion



# def main():
#     args = sys.argv[1:]

#     if len(args) == 3:
#         if args[0] == 'nav': env_name = 'nav'
#         if args[0] == 'blocks': env_name = 'blocks_operator_actions'
#         if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
#         templateFile = args[1]
#         goalsFile = args[2]

#         dir = templateFile.rpartition('/')[0]
#         problem_dir = f'{dir}/problem_files'
#         results_dir = f'{dir}/RESULTS'

#         if not os.path.exists(problem_dir): os.makedirs(problem_dir)
#         if not os.path.exists(results_dir): os.makedirs(results_dir)


#         with open(goalsFile) as file:
#             goals = file.readlines()
        

#         for i, goal in enumerate(goals):
#             print(goal)

#             with open(templateFile, 'r') as file:
#                 filedata = file.read()

#             filedata = filedata.replace('<HYPOTHESIS>', goal)

#             with open(f'{problem_dir}/template_temp.pddl', 'w') as file:
#                 file.write(filedata)
            
#             env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
#             env.fix_problem_index(0)
              
            
#             learn(env, results_dir, i)

