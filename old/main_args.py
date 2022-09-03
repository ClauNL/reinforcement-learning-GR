import os
import matplotlib
from matplotlib.pyplot import jet
from pddlgym.rendering import blocks, hanoi; matplotlib.use('agg')
import pddlgym
import joblib
import sys, getopt

from pathlib import Path  
file = Path(__file__).resolve()  
directory = file.parents[0]  

from myfunctions import *
# from main import *

# python3 baseline2.py -d Blocks -p 0 -t -s

# java -jar goalrecognizer1.0.jar -filter experiments/blocks-test/blocks-test.tar.bz2 0
# java -jar goalrecognizer1.0.jar -goalcompletion experiments/blocks-test/domain.pddl experiments/blocks-test/template.pddl experiments/blocks-test/hyps.dat experiments/blocks-test/obs.dat experiments/blocks-test/real_hyp.dat 0.1

# python3 main_args.py domain.ppdl template.pddl hyps.dat real_hyp.dat
# python3 main_args.py 'blocks' template.pddl hyps.dat real_hyp.dat


# test_files
#     blocks
#         problem0
#             template
#             file0
#             file1
#             file2

#     hanoi
#     nav



  
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
        

        print(f'goal {goal}: {dist}' )

        if dist <= min_dist:
            min_dist = dist
            predicted_goal = goal

    # print(f'predicted goal: {predicted_goal}')
    return predicted_goal



def learn(env, results_dir, goal):

    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    q = qLearning(env, q, num_episodes = 200, max_steps = 10000)  # Q Learning
    writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q
    #video(env, q, q_name, results_dir)    # crear video con la solucion

def show_video(env, results_dir, start, goal):
    q_name = f'q_g{goal}'
    q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    video(env, q, f's{start}_g{goal}', results_dir)

# domain  template  goals  init
def main():
    args = sys.argv[1:]

    if len(args) == 3:
        if args[0] == 'nav': env_name = 'nav'
        if args[0] == 'blocks': env_name = 'blocks_operator_actions'
        if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
        templateFile = args[1]
        goalsFile = args[2]
        # initFile = args[3]

        dir = templateFile.rpartition('/')[0]
        problem_dir = f'{dir}/problem_files'
        results_dir = f'{dir}/RESULTS'

        if not os.path.exists(problem_dir): os.makedirs(problem_dir)
        if not os.path.exists(results_dir): os.makedirs(results_dir)


        with open(goalsFile) as file:
            goals = file.readlines()
        # with open(initFile) as file:
        #     starts = file.readlines()
        

        # --------  LEARN  ----------
        # for i, goal in enumerate(goals):
        #     print(goal)

        #     with open(templateFile, 'r') as file:
        #         filedata = file.read()

        #     filedata = filedata.replace('<HYPOTHESIS>', goal)
        #     # filedata = filedata.replace('<START>', starts[0])

        #     with open(f'{problem_dir}/template_g{i}.pddl', 'w') as file:
        #         file.write(filedata)
            
        #     env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
        #     env.fix_problem_index(i)
              
        #     learn(env, results_dir, i)


        # --------  VIDEO  ----------
        # for i, start in enumerate(starts):
        #     for j, goal in enumerate(goals):
        #         with open(templateFile, 'r') as file:
        #             filedata = file.read()

        #         filedata = filedata.replace('<HYPOTHESIS>', goal)
        #         filedata = filedata.replace('<START>', start)

        #         with open(f'{problem_dir}/template_temp.pddl', 'w') as file:
        #             file.write(filedata)
            
        #         env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
        #         env.fix_problem_index(0)

        #         show_video(env, results_dir, i,j)

        # --------  VIDEO  ----------
        # for i, goal in enumerate(goals):
           
        #     env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
        #     env.fix_problem_index(i)

        #     q_name = f'q_g{i}'
        #     q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
        #     video(env, q, f'g{i}', results_dir)




        # sum = 0
        # for i, start in enumerate(starts):
        #     if i != 2: continue
        #     for j, goal in enumerate(goals):
        #         with open(templateFile, 'r') as file:
        #             filedata = file.read()

        #         filedata = filedata.replace('<HYPOTHESIS>', goal)
        #         filedata = filedata.replace('<START>', start)

        #         with open(f'{problem_dir}/template_temp.pddl', 'w') as file:
        #             file.write(filedata)
            
        #         env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
        #         env.fix_problem_index(0)


        #         print(f'selected goal: {j}')

        #         obs = get_observation(env, results_dir, j)
        #         # for i in range(0, len(obs), 2):
        #         #     print(obs[i+1])
        #         predicted_goal = goalRecognition(obs, 'MaxUtil',  results_dir, len(goals))
        #         print(f'predicted goal: {predicted_goal}\n')

        #         if j == predicted_goal:
        #             sum += 1
        
        # print(sum)

        sum = 0
     
        for j, goal in enumerate(goals):
            
            env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
            env.fix_problem_index(j)

            print(f'selected goal: {j}')

            obs = get_observation(env, results_dir, j)
            # for i in range(0, len(obs), 2):
            #     print(obs[i+1])
            predicted_goal = goalRecognition(obs, 'DP',  results_dir, len(goals))
            print(f'predicted goal: {predicted_goal}\n')

            if j == predicted_goal:
                sum += 1
        
        print(sum)





          


                


main()


# python3 main_args.py 'nav' test_files/nav/problem2/template.pddl test_files/nav/problem2/goals.dat











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

