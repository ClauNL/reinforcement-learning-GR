import enum
import math
import os
import pickle
import random
import matplotlib

from pddlgym.core import InvalidAction, PDDLEnv
from pddlgym_planners.fd import FD; matplotlib.use('agg')

import pddlgym

import imageio
import numpy as np
import joblib

import csv






def qLearning(env, q = {}, 
                num_episodes = 200,
                max_steps = 100,

                alpha = 0.1,
                discount = 0.4,

                epsilon = 1,
                max_epsilon = 1,
                min_epsilon = 0.01,
                epsilon_decay_rate = 0.001 ):

    
    rewards = []

    for episode in range(num_episodes):
        print(episode)
        obs, _ = env.reset()
        rewards_current_episode = 0
        done = False

        for step in range(max_steps): 

            # explotación
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > epsilon:

                if obs.literals in q:
                    action = max(q[obs.literals], key = q[obs.literals].get) 
                else:
                    action = env.action_space.sample(obs)

                
            # exploración
            else:
                action = env.action_space.sample(obs)

            
            # ejecutar action
            new_obs, reward, done, _ = env.step(action)

            
            if obs.literals not in q:
                q[obs.literals] = {}
            
            if new_obs.literals in q:
                max_next_state = max(q[new_obs.literals].values())
            else:
                max_next_state = 0.0

            # actualizar tabla Q
            q[obs.literals][action] = (1-alpha) * q[obs.literals].get(action, 0.0) + alpha * (reward + discount * max_next_state )

        
            # actualizar estado
            obs = new_obs
            #state = new_state

            rewards_current_episode += reward 

            if done == True: 
                break

        # Exploration rate decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate*episode)

        rewards.append(rewards_current_episode)

    print(len(q))

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards),num_episodes/100)
    count = 100

    print("********Average reward********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/100)))
        count += 100
    
    return q




def qLearning2(env, action_list, q = {}, 
                num_episodes = 500,
                max_steps = 100,

                alpha = 0.1,
                discount = 0.4,

                epsilon = 1,
                max_epsilon = 1,
                min_epsilon = 0.01,
                epsilon_decay_rate = 0.001):

    
    rewards = []
    num_actions = len(action_list)

    for episode in range(num_episodes):
        # print(episode)
        obs, _ = env.reset()
        state = tuple(sorted(tuple(obs.literals)))
        rewards_current_episode = 0
        done = False

        for step in range(max_steps): 

            # explotación
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > epsilon:

                if state in q:
                    action = np.argmax(q[state])
                    # action = action_list[action_num]
                    # action = max(q[obs.literals], key = q[obs.literals].get) 
                else:
                    selected_action = env.action_space.sample(obs)
                    # action = random.randint(0, num_actions-1)
                    action = action_list.index(selected_action)


            # exploración
            else:
                selected_action = env.action_space.sample(obs)
                # action = random.randint(0, num_actions-1)
                action = action_list.index(selected_action)

            # ejecutar action
            # print(f'ACTION:  {action} - {action_list[action]}')
            
            try:
                new_obs, reward, done, _ = env.step(action_list[action])
                new_state = tuple(sorted(tuple(new_obs.literals)))
            except InvalidAction:
                new_obs = obs
                new_state = state
                reward = 0
                done = False
                
            


            
            # new_obs, reward, done, _ = env.step(action_list[action])
            # new_state = tuple(sorted(tuple(new_obs.literals)))

            

            if state not in q:
                # q[state] = {}
                q[state] = [0.]*len(action_list)
            
            if new_state in q:
                # max_next_state = max(q[new_obs.literals].values())
                max_next_state = np.max(q[new_state])
            else:
                max_next_state = 0.0

            # actualizar tabla Q
            q[state][action] = (1-alpha) * q[state][action] + alpha * (reward + discount * max_next_state )

        
            # actualizar estado
            obs = new_obs
            state = new_state

            rewards_current_episode += reward 

            if done == True: 
                # print('done')
                break

        # Exploration rate decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate*episode)

        rewards.append(rewards_current_episode)

    print(len(q))

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards),num_episodes/100)
    count = 100

    print("********Average reward********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/100)))
        count += 100


    return q, sum(rewards_per_thousand_episodes[-1]/100)
    


def readQtable(path):
    q = {}
    if os.path.exists(path):
        q = joblib.load(path)
    return q



def writeQtable(q, name, path):
    joblib.dump(q, f'{path}/{name}.pkl')
    w = csv.writer(open(f'{path}/{name}.csv', "w"))
    for key, val in q.items():
        w.writerow([key, val])




def create_video(env, q, name, path, verbose = False):

    images = []

    # initialize new episode params
    obs, _ = env.reset()
    #state = frozenset({str(i) for i in obs.literals})
    done = False

    start = env.render()
    end = env.render()

    for step in range(500):        
        # Show current state of environment on screen
        images.append(env.render())
        
        # Choose action with highest Q-value for current state      
        if obs.literals in q:
            action = max(q[obs.literals], key = q[obs.literals].get) 
        else:
            action = env.action_space.sample(obs)  

        if verbose:
            #print(state)
            print(action)  
                
        # Take new action
        obs, reward, done, debug_info = env.step(action)
        #state = frozenset({str(i) for i in obs.literals})
    
        if done:
            images.append(env.render()) 
            end = env.render()                   
            break     


    imageio.imsave(f'{path}/start.png', start)
    imageio.imsave(f'{path}/end_{name}.png', end)
    imageio.mimwrite(f'{path}/video_{name}.mp4', images, fps=3)
    print('Wrote out video')


    #return start, end, images



def create_video2(env, q, action_list, name, path, verbose = False):

    images = []

    # initialize new episode params
    obs, _ = env.reset()
    state = tuple(sorted(tuple(obs.literals)))
    done = False

    start = env.render()
    end = env.render()

    for step in range(500):        
        # Show current state of environment on screen
        images.append(env.render())
        
        # Choose action with highest Q-value for current state 
        if state in q:
            action = np.argmax(q[state]) 
        else:
            selected_action = env.action_space.sample(obs)
            action = action_list.index(selected_action)   
 

        if verbose:
            print(action_list[action])  
                
        # Take new action
        try:
            obs, reward, done, _ = env.step(action_list[action])
            state = tuple(sorted(tuple(obs.literals)))
        except InvalidAction:
            done = False
    
        if done:
            images.append(env.render()) 
            end = env.render()                   
            break     


    if not os.path.exists(path): os.makedirs(path)
    imageio.imsave(f'{path}/start.png', start)
    imageio.imsave(f'{path}/end_{name}.png', end)
    imageio.mimwrite(f'{path}/video_{name}.mp4', images, fps=3)
    print('Wrote out video')




def create_video_from_obs(env, observation, path):

    images = []

    # initialize new episode params
    obs, _ = env.reset()
    done = False
    
    for state, action in observation:
        images.append(env.render())
        obs, reward, done, _ = env.step(action)

    images.append(env.render())
  

    if not os.path.exists(path): os.makedirs(path)
    imageio.mimwrite(f'{path}/video_OBS.mp4', images, fps=3)
    print('Wrote out video')


def create_video_from_obs_goals(env, observation, path, goals_list):

    images = []

    # initialize new episode params
    obs, _ = env.reset()
    done = False
    
    for state, action in observation:
        images.append(env.render(mode='goals_layout', goals = goals_list))
        obs, reward, done, _ = env.step(action)

    images.append(env.render(mode='goals_layout', goals = goals_list))
  

    if not os.path.exists(path): os.makedirs(path)
    imageio.mimwrite(f'{path}/video_OBS.mp4', images, fps=3)
    print('Wrote out video')



def create_observabilities2(d, output, actions, ind=0):
    print(output + '/problems')
    print(d + "/domain.pddl")
    env = PDDLEnv(d + "/domain.pddl", d + '/problems', raise_error_on_invalid_action=False,
                  dynamic_action_space=True)
    
    env.fix_problem_index(ind)
    init, _ = env.reset()
    print(init.goal)
    #print(f'GOAL {init.goal}')
    
    # traj is an action pair tuple, need to map this to state action number pair
    traj = []
    obs_list = [1.0]
    traj_list = {}
    for a in actions:
        state_action_pair = (solve_fset(init.literals), a)
        traj.append(state_action_pair)
        init, _, _, _ = env.step(a)
    print(actions)
    
    for obs in obs_list:
        traj_list[obs] = remove_obs(traj, obs)
        save_obs(traj_list[obs], output + '/' + 'obs' + str(obs)+'.dat')
        with open(output + '/' + 'obs' + str(obs) + '.pkl', "wb") as output_file:
            pickle.dump(traj_list[obs], output_file)


def create_observabilities(d, output, ind=0):
    print(output + '/problems')
    print(d + "/domain.pddl")
    env = PDDLEnv(d + "/domain.pddl", d + '/problems', raise_error_on_invalid_action=False,
                  dynamic_action_space=True)
    planner = FD()
    env.fix_problem_index(ind)
    init, _ = env.reset()
    print(init.goal)
    #print(f'GOAL {init.goal}')
    
    # traj is an action pair tuple, need to map this to state action number pair
    plan = planner(env.domain, init)
    traj = []
    obs_list = [1.0]
    traj_list = {}
    for a in plan:
        state_action_pair = (solve_fset(init.literals), a)
        traj.append(state_action_pair)
        init, _, _, _ = env.step(a)
    print(plan)
    
    for obs in obs_list:
        traj_list[obs] = remove_obs(traj, obs)
        save_obs(traj_list[obs], output + '/' + 'obs' + str(obs)+'.dat')
        with open(output + '/' + 'obs' + str(obs) + '.pkl', "wb") as output_file:
            pickle.dump(traj_list[obs], output_file)

def save_obs(traj, out):
    new_obs = open(out, "w")
    for line in traj:
        str_out = ''
        for pred in line[0]:
            str_out += str(pred) + ' '
        str_out += ';'
        str_out += str(line[1])
        new_obs.write(str_out)
        new_obs.write('\n')
    new_obs.close()


def remove_obs(instance, observability):
    new_obs = []
    n_observations = len(instance)

    # Number of observations to remove
    n_remove = int(n_observations*(1-observability))

    # Randomly sample indices to remove from the states list
    indices = sorted(random.sample(range(0, n_observations), n_remove))

    # Create new list with states except the indices to remove
    for i in range(n_observations):
        if i not in indices:
            new_obs.append(instance[i])
    return new_obs


def solve_fset(fset):
    '''
    Converts a fronzenset to an ordered tuple.
    '''
    return tuple(sorted(tuple(fset)))


def getObservation(env, q, steps):

    observation = []
    obs, _ = env.reset()
    done = False

    for step in range(steps):        
        
        # Choose action with highest Q-value for current state      
        if obs.literals in q:
            action = max(q[obs.literals], key = q[obs.literals].get) 
        else:
            action = env.action_space.sample(obs)  

        observation.append(obs.literals)
        observation.append(action)
                
        # Take new action
        obs, reward, done, debug_info = env.step(action)
    
        if done:                  
            break  

    return observation











def MaxUtil(q, observation):
    sum = 0
    for i in range(0, len(observation), 2):
        state = observation[i]
        action = observation[i+1]
        if state in q:
            sum += q[state].get(action, 0.0)

    return -sum


def policy(q, a, s):
    p = 0
    if s in q:
        val1 = q[s].get(a, 0.0) 
        val2 = sum(q[s].values())
        if val2 != 0:
            p = val1 / val2
    return p


def KL(q, observation):
    sum = 0
    for i in range(0, len(observation), 2):
        state = observation[i]
        action = observation[i+1]
        # print(action)

        pg = policy(q, action, state)

        print(f'pg = {pg}')

        if pg != 0:
            print(f'log = {math.log(pg)}')
            sum += (pg * math.log(pg))
            print(f'sum = {pg * math.log(pg)}')

    return - sum

  



def DP(q, observation):
    delta = 0.3
    min_t = math.ceil(len(observation)/2)
    # print(f'min t: {min_t}')
  
    for i in range(0, len(observation), 2):
        # print(observation[i+1])
        if i == 0:
            continue
        real_i = math.ceil(i/2)
        prev_state = observation[i-2]
        prev_action = observation[i-1]

        
        p = policy(q, prev_action, prev_state)

        # print(f'i: {real_i} ')
        # print(f'p: {p} ')

        if p <= delta:
            if real_i < min_t:
                min_t = real_i
                     
    return - min_t






def MaxUtil2(q, actions, observation):
    sum = 0
    # print('alo')
    for state, action in observation:
        if state in q:
            # print('estoy')
            # print(q[state][actions.index(action)])
            # print(q[state])
            sum += q[state][actions.index(action)]
            # print(q[state][actions.index(action)])

    return -sum


def policy2(q_values):
    result = q_values
    total = sum(q_values)
    if total != 0:
        result = [value / total for value in q_values]

    return result

def KL2(q, actions, observation):
    sum = 0
    distances = []

    for state, action in observation:
        action_index = actions.index(action)
        
            
        # if state in q:
        # print('EDTABA')
        # pg = policy2(q[state], action_index) 
        if state not in q:
            pg = [1./len(actions) for _ in range(len(actions))]
            # print(pg)
        else:
            # pg = policy2(q[state])[action_index]
            pg = policy2(q[state])
        # po = 1
        po = [0 for _ in range(len(actions))]
        po[action_index] = 1. 
        if pg[action_index] != 0:
   
        # distances.append(kl_divergence(po, pg))
            distances.append(po[action_index] * math.log2(po[action_index]/pg[action_index]))


       
        
    # print(distances)
    if not distances:
        print('NOT DISTANCES')
    return np.mean(distances)


def kl_divergence(p1, p2):

    assert(len(p1) == len(p2))
    # print(len(p1))
    # print(len(p2))
    # return sum(p1[i] * math.log2(p1[i]/p2[i]) for i in range(len(p1)))
    sum = 0
    for i in range(len(p1)):
        if p2[i] != 0:
           
            sum += p1[i] * math.log2(p1[i]/p2[i])
        # print(p1[i] * math.log2(p1[i]/p2[i]))
        # print(p1[i])
        # print(p2[i])
    # print(sum)
    return sum
  
  
def kl_divergence_norm_softmax(pol, actions, traj):
  
    p_traj = traj_to_policy(traj, actions)

    softmax_policy = {state: policy2(pol[state]) for state in pol.keys()}
    distances = []
    for i, state in enumerate(p_traj):
        if state not in softmax_policy:
            add_dummy_policy(softmax_policy, state, actions)
        qp1 = p_traj[state]
        qp2 = softmax_policy[state]
        if qp2 == 0:
            print('ES CEROOOOOO')
        # print(qp1)
        # print(qp2)
        distances.append(kl_divergence(qp1, qp2))
    return np.mean(distances)
    # return 0

def traj_to_policy(trajectory, actions, epsilon: float = 0.):
    # converts a trajectory from a planner to a policy
    # where the taken action has 99.99999% probability
    trajectory_as_policy = {}
    for state, action in trajectory:
        action_index = actions.index(action)
        actions_len = len(actions)
        qs = [1e-6 + epsilon/actions_len for _ in range(actions_len)]
        qs[action_index] = 1. - 1e-6 * (actions_len-1) - epsilon
        trajectory_as_policy[tuple(state)] = qs
    return trajectory_as_policy


def add_dummy_policy(softmax_policy, state, actions):
    # returns a dummy behavior in case a state has not been visited
    # when running a tabular policy
    n_actions = len(actions)
    softmax_policy[state] = [1./n_actions for _ in range(n_actions)]

# def policy2(q_values, action_index):
#     p = 0
#     val1 = q_values[action_index]
#     val2 = sum(q_values)
#     if val2 != 0:
#         p = val1 / val2
#     return p



def DP2(q, actions, observation):
    delta = 0.4

    for i, (state, action) in enumerate(observation):
        # print('entro')
        action_index = actions.index(action)
        p = 0
        if state in q:
            # print('estoy')
            
            # p = policy2(q[state], action_index) 
            p = policy2(q[state])[action_index]
            # print(p)
        if p < delta:
            return -i 

    return -len(observation)


  