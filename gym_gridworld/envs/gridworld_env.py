import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
import random
from pdb import set_trace
import pickle

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[1.0,0.0,0.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0], 8: [1.0,0.0,0.0],
          9:[1.0,0.0,0.0], 10: [1.0,0.0,0.0]}

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0 
    def __init__(self):
        self._seed = 0
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
        self.agent_start_locs = [[1,1], [1,7], [7,1], [7,7]]
        self.trial_counter = 0
        self.steps = []
        self.step_counter = 0
        self.player = 'None'
        #self.num_plays = 0
 
        ''' set observation space '''
        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
    
        ''' initialize system state ''' 
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan' + str(random.randint(0,9)) + '.txt')     
        self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map

        ''' Reset agent location '''
        new_agent_loc = self.agent_start_locs[random.randint(0,3)]
        self.start_grid_map[new_agent_loc[0], new_agent_loc[1]] = 4

        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state(self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self.verbose = False # to show the environment or not

        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env 
        if self.verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self._render()

    def step(self, action):
        self.trial_counter +=1
        self.step_counter += 1
        
        ''' return next observation, reward, finished, success '''
        action = int(action)
        info = {}
        info['success'] = False
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                            self.agent_state[1] + self.action_pos_dict[action][1])
        #self.agent_states = np.transpose(np.nonzero((self.current_grid_map == 8) | (self.current_grid_map == 4)))
        #self.available_states = np.transpose(np.nonzero(self.current_grid_map == 0))
        if action == 0: # stay in place
            info['success'] = True
            return (self.observation, 0, False, info) 
        if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
            info['success'] = False
            return (self.observation, 0, False, info)
        if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
            info['success'] = False
            return (self.observation, 0, False, info)
        # successful behavior
        org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
        new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        if new_color == 0:
            if org_color == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
            elif org_color == 6 or org_color == 7:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color-4 
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        elif new_color == 1: # gray
            info['success'] = False
            return (self.observation, 0, False, info)
        elif new_color == 2 or new_color == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self._render()
        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] :
            target_observation = copy.deepcopy(self.observation)
            if self.restart_once_done:
                self.observation = self.reset()
                info['success'] = True
                return (self.observation, 1, True, info)
            else:
                info['success'] = True
                return (target_observation, 1, True, info)
        else:
            info['success'] = True
            return (self.observation, 0, False, info)

    def reset(self):        
        self.steps.append(self.step_counter)
        #self.exp_neptune.log_metric('steps', self.step_counter)
        self.step_counter = 0
        if self.verbose == True:
            print('steps array: ', self.steps)
            print('trial counter: ', self.trial_counter)
        
        #save data after n games completed
        if len(self.steps) % 100 == 0:
            #self.num_plays += 1
            #with open('data/' + self.player + '_' + str(self.num_plays) + '.pkl', 'wb') as f:
            with open('data/' + self.player + '_' + str(self.exp_name) + '.pkl', 'wb') as f:
                pickle.dump(self.steps[2:], f)
                set_trace()

        ''' Reset grid state and agent location '''
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan' + str(random.randint(0,9)) + '.txt')      
        self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map

        new_agent_loc = self.agent_start_locs[random.randint(0,3)]
        self.start_grid_map[new_agent_loc[0], new_agent_loc[1]] = 4
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state(self.start_grid_map)
        
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        #     grid_map_array = []
        #     for row in grid_map:
        #         result = [int(x) for x in row.split(' ')]
        #         grid_map_array.append(result)
        # grid_map_array = np.reshape(grid_map_array, (len(grid_map_array[0]), len(grid_map_array[0])))
        grid_map_array = np.array(
           list(map(
               lambda x: list(map(
                   lambda y: int(y),
                   x.split(' ')
               )),
              grid_map
           ))
        )
        return grid_map_array

    # def _get_game_state(self):
    #     '''
    #     Return whether the level has ended.
    #     '''
    #     return self.reset

    def _get_agent_start_target_state(self, start_grid_map):
        '''
        Return agent (=4) starting location and current goal (=3) location. If 4 and 3 dont' exist, throw error. 
        '''

        start_state = None
        target_state = None
        agent_states = []
        start_state = list(map(
            lambda x:x[0] if len(x) > 0 else None,
            np.where(start_grid_map == 4)
        ))
        target_state = list(map(
            lambda x:x[0] if len(x) > 0 else None,
            np.where(start_grid_map == 3)
        ))
        self.agent_states = np.transpose(np.nonzero((start_grid_map == 8) | (start_grid_map == 4)))
        self.available_states = np.transpose(np.nonzero(start_grid_map == 0))

        if start_state == [None, None] or target_state == [None, None]:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1] = np.array(COLORS[grid_map[i,j]])
        return observation
  
    def _render(self, mode='human', close=False):
        if self.verbose == False:
            return
        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self.reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 0
            self.start_grid_map[sp[0], sp[1]] = 4
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.reset()
            self._render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self.reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != 0:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = 0
            self.start_grid_map[tg[0], tg[1]] = 3
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.reset()
            self._render()
        return True

    def get_grid_state(self):
        ''' get current grid state '''
        return self.current_grid_map, self.available_states, self.agent_states, self.agent_target_state, self.agent_state
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        info = {}
        info['success'] = True
        if self.current_grid_map[to_state[0], to_state[1]] == 0:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 4:
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 1:
            info['success'] = False
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._render()
            if self.restart_once_done:
                self.observation = self.reset()
                return (self.observation, 1, True, info)
            return (self.observation, 1, True, info)
        else:
            info['success'] = False
            return (self.observation, 0, False, info)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 
