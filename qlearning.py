import glob
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



#  domain  dir
#  domain  dir   < -learn | -video >
#  domain  dir  goal
#  domain  dir  goal < -learn | -video >  

def main():
    args = sys.argv[1:]

    if len(args) != 2 & len(args) != 3 & len(args) != 4: exit()

    learn = False
    video = False
    selected_goal = -1


    if args[0] == 'nav': env_name = 'nav'
    if args[0] == 'blocks': env_name = 'blocks_operator_actions'
    if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
    dir = args[1]

    if len(args) == 2:
        learn = True
        video = True

    elif len(args) == 3:
        if args[2] == '-learn': learn = True
        elif args[2] == '-video': video = True
        else: 
            selected_goal = args[2]
            learn = True
            video = True

    elif len(args) == 4:
        selected_goal = args[2]
        if args[3] == '-learn': learn = True
        if args[3] == '-video': video = True

    
    templateFile = f'{dir}/template.pddl'
    goalsFile = f'{dir}/goals.dat'
    startsFile = f'{dir}/starts.dat'

    problem_dir = f'{dir}/problem_files'
    results_dir = f'{dir}/RESULTS'

    if not os.path.exists(problem_dir): os.makedirs(problem_dir)
    if not os.path.exists(results_dir): os.makedirs(results_dir)


    with open(goalsFile) as file:
        goals = file.readlines()
    with open(startsFile) as file:
        starts = file.readlines()
    


    for i, start in enumerate(starts):
        for j, goal in enumerate(goals):
            with open(templateFile, 'r') as file:
                filedata = file.read()

            filedata = filedata.replace('<HYPOTHESIS>', goal)
            filedata = filedata.replace('<START>', start)

            with open(f'{problem_dir}/template_s{i}_g{j}.pddl', 'w') as file:
                file.write(filedata)



    # --------  LEARN  ----------
    if learn:
        if selected_goal == -1:
            for i in range(len(goals)):
                
                env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
                env.fix_problem_index(i)
                    
                q_name = f'q_g{i}'
                q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
                q = qLearning(env, q, num_episodes = 500, max_steps = 10000)  # Q Learning
                writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q

        else:
            env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
            env.fix_problem_index(int(selected_goal))
                
            q_name = f'q_g{selected_goal}'
            q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
            q = qLearning(env, q, num_episodes = 500, max_steps = 10000)  # Q Learning
            writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q


    # --------  VIDEO  ----------
    if video:
        if selected_goal == -1:
            for i, start in enumerate(starts):
                for j, goal in enumerate(goals):

                    env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
                    print(len(goals)*i + j)
                    env.fix_problem_index(len(goals)*i + j)

                    q = readQtable(f'{results_dir}/q_g{j}.pkl') # cargar tabla Q si existe
                    create_video(env, q, f's{i}_g{j}', results_dir)

        else:
            for i, start in enumerate(starts):
                env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
                print(len(goals)*i + int(selected_goal))
                env.fix_problem_index(len(goals)*i + int(selected_goal))

                q = readQtable(f'{results_dir}/q_g{int(selected_goal)}.pkl') # cargar tabla Q si existe
                create_video(env, q, f's{i}_g{int(selected_goal)}', results_dir)




main()

# python3 qlearning.py -learn 'nav' test_files/nav/problem2

# python3 qlearning.py -learn 'nav' test_files/nav/problem2/template.pddl test_files/nav/problem2/goals.dat test_files/nav/problem2/starts.dat
# python3 qlearning.py -video 'nav' test_files/nav/problem2/template.pddl test_files/nav/problem2/goals.dat test_files/nav/problem2/starts.dat
# python3 qlearning.py 'nav' test_files/nav/problem2/template.pddl test_files/nav/problem2/goals.dat test_files/nav/problem2/starts.dat






    # --------  LEARN  ----------
    # if learn:
    #     for i, goal in enumerate(goals):
    #         # print(goal)

    #         with open(templateFile, 'r') as file:
    #             filedata = file.read()

    #         filedata = filedata.replace('<HYPOTHESIS>', goal)
    #         filedata = filedata.replace('<START>', starts[0])

    #         with open(f'{problem_dir}/template_g{i}.pddl', 'w') as file:
    #             file.write(filedata)
            
    #         env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
    #         env.fix_problem_index(i)
                
    #         q_name = f'q_g{i}'
    #         q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    #         q = qLearning(env, q, num_episodes = 500, max_steps = 10000)  # Q Learning
    #         writeQtable(q, f'{q_name}', results_dir)  # guardar tabla Q






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
    # if video:
    #     for i, goal in enumerate(goals):
            
    #         env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0", problem_dir = problem_dir) 
    #         env.fix_problem_index(i)

    #         q_name = f'q_g{i}'
    #         q = readQtable(f'{results_dir}/{q_name}.pkl') # cargar tabla Q si existe
    #         create_video(env, q, f'g{i}', results_dir)







# elif len(args) == 4:
#         learn = True
#         video = True
#         if args[0] == 'nav': env_name = 'nav'
#         if args[0] == 'blocks': env_name = 'blocks_operator_actions'
#         if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
#         templateFile = args[1]
#         goalsFile = args[2]
#         startsFile = args[3]
#         dir = templateFile.rpartition('/')[0]

#     elif len(args) == 5:
#         if args[0] == '-learn': learn = True
#         if args[0] == '-video': video = True
#         if args[1] == 'nav': env_name = 'nav'
#         if args[1] == 'blocks': env_name = 'blocks_operator_actions'
#         if args[1] == 'hanoi': env_name = 'hanoi_operator_actions'
#         templateFile = args[2]
#         goalsFile = args[3]
#         startsFile = args[4]
#         dir = templateFile.rpartition('/')[0]






    # if len(args) == 2:
    #     learn = True
    #     video = True
    #     if args[0] == 'nav': env_name = 'nav'
    #     if args[0] == 'blocks': env_name = 'blocks_operator_actions'
    #     if args[0] == 'hanoi': env_name = 'hanoi_operator_actions'
    #     dir = args[1]
        
    # elif len(args) == 3:
    #     if args[0] == '-learn': learn = True
    #     if args[0] == '-video': video = True
    #     if args[1] == 'nav': env_name = 'nav'
    #     if args[1] == 'blocks': env_name = 'blocks_operator_actions'
    #     if args[1] == 'hanoi': env_name = 'hanoi_operator_actions'
    #     dir = args[2]

    # elif len(args) == 4:
    #     if args[0] == '-learn': learn = True
    #     if args[0] == '-video': video = True
    #     if args[1] == 'nav': env_name = 'nav'
    #     if args[1] == 'blocks': env_name = 'blocks_operator_actions'
    #     if args[1] == 'hanoi': env_name = 'hanoi_operator_actions'
    #     dir = args[2]
    #     selected_goal = args[3]
    