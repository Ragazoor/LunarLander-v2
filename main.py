import numpy as np
import matplotlib.pyplot as plt
import logging
import gym
from gym.wrappers import Monitor
from FFNNAgent import *
#from tensorflow.python.client import device_lib
###########
# STRUCTURE
###########

# MAIN - Initialises FFNNAgent, trains agent, 
#   |
#   |
# FFNNAgent - Recieves feedBack from env, makes action
#   |
#   |
# FFNN - called by Agent for deciding action and training

def main_FFNNAgent():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Init and constants:
    
    env = gym.make('LunarLander-v2')
    env = Monitor(directory='tmp/LunarLander-v2',video_callable=False, force = True, write_upon_reset=True)(env)
#    env = gym.wrappers.Monitor('LunarLander-v2', outdir)

    print env.observation_space
    print env.observation_space.high
    print env.observation_space.low
    print env.action_space
    
#    print get_available_gpus()

    # Hyperparams:  learning options, network structure, number iterations & steps,
    hyperparams = {}
    # ----------- Net Parameters:
    hyperparams['gamma'] = 1  #0.98
    hyperparams['n_input_nodes'] = 8
    hyperparams['n_hidden_nodes'] = 20 #30
    hyperparams['n_output_nodes'] = 4
    hyperparams['n_steps'] = 1000
    hyperparams['seed'] = 14  # 13
    # ------------------------------------------------------
    hyperparams['init_net_wr'] = 0.05  # 0.05
    hyperparams['batch_size'] = 50  # 400
    hyperparams['epsilon'] = 1  # 1 - starting value
    hyperparams['epsilon_min'] = 0.2  # 0 - Need to explore alot so it doesn't stick in local max
    hyperparams['epsilon_decay_rate'] = 0.995  # 995  
#   ~.99 over 200 leaves it 0.1339 ~.995 over 500 its leaves it at 0.08
 
   # ----------------------------------------------------------
    hyperparams['target_net_hold_epsiodes'] = 1  # 6
    hyperparams['learning_rate'] = 0.05 # 0.05
    hyperparams['learning_rate_min'] = 0  #0.01 # 11 or 0.01
    hyperparams['learning_rate_decay'] = 0.5  # 0.5
    hyperparams['n_updates_per_episode'] = 1  # 1 - means pick X random minibatches, doing GradDescent on each
    hyperparams['nmr_decimals_tiles'] = 3 # the resolution of the tiles are 10^-x
    hyperparams['max_memory_len'] = 5000  # 1000 - number of (s,a,r,s',done) tuples
    hyperparams['n_iter'] = 2000  # 1000
    hyperparams['n_episodes_per_print'] = 50
    hyperparams['net_hold_epsilon'] = 2 # 3
    hyperparams['net_hold_lr'] = 30000000 # 2000
    hyperparams['C'] = 0 # Higher values encourages exploration
    # ------------ BEST SETTINGS GIVE: test mean:  +- 0
    # FFNN agent:
    agent = FFNNAgent(hyperparams)            

    # starts to train agent
    agent.optimize_episodes(env, rend = True)
    agent.net.plot_error()    
    agent.plot_reward()
    plt.show()

    # test to see how it goes
    agent.epsilon = 0
    agent.n_iter = 100
    agent.n_episodes_per_print = 5
    agent.C = 0
    agent.learning_rate = 0.01
    agent.net_hold_lr = 1
    agent.learning_rate_decay = 1
    agent.optimize_episodes(env, rend = True)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos ]

if __name__ == '__main__':
    main_FFNNAgent()


