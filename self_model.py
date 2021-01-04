from pdb import set_trace
import numpy as np
from utils.keys import *
import copy

class Self_class():
    def __init__(self):
        self.me = None
        self.tried_keys = []
        self.keys = [0, 1, 2, 3] #U, D, L, R
        self.last_grid = []
        self.last_action = None
    
    def predict(self, env):
        if env.game_type == 'logic':
            action = self.predict_logic(env)
        if env.game_type == 'contingency':
            action = self.predict_contingency(env)
        return action

    def predict_logic(self, env):
        # Get state
        grid, avail, agents, target, SELF = env.get_grid_state()
        around = [[],[],[],[]]
        around_clear = [[],[],[],[]]

        # (1) If any avatar is closer to the target, move that one
        #distances = [np.sqrt((target[1]-arr1[1])**2 + (target[0]-arr1[1])**2) for arr1 in agents] #get euclideant distance

        #distance = np.sqrt((target[1]-SELF[1])**2 + (target[0]-SELF[0])**2) #get euclidean distance

        if(len(self.last_grid) > 0):
            diff = grid - self.last_grid
            if(sum(diff.flatten() != 0)):
                print('navigating the self!')
                dist_vertical = target[0] - SELF[0]
                dist_horiz = target[1] - SELF[1]

                if dist_vertical > 0:
                    return key_converter(1)
                elif dist_vertical < 0:
                    return key_converter(0)
                elif dist_horiz > 0:
                    if dist_horiz == 1:
                        self.last_grid = []
                        self.tried_keys = []
                    return (key_converter(3))
                elif dist_horiz < 0:
                    if dist_horiz == -1:
                        self.last_grid = []
                        self.tried_keys = []
                    return (key_converter(2))

        # (2) ...else, pick the option most likely to localize the self
        print('FINDING THE SELF!')

        # Get indeces of blocks around agents (regardless of their contents)
        for i, agent in enumerate(agents):
            around[i].append(grid[agent[0]-1, agent[1]])
            around[i].append(grid[agent[0]+1, agent[1]])
            around[i].append(grid[agent[0], agent[1]-1])
            around[i].append(grid[agent[0], agent[1]+1])
        around = np.array(around) 
        zeros = np.transpose(np.nonzero(around == 0)) #which blocks have zeros
        unique, counts = np.unique(zeros[:,1], return_counts=True) #which available direction is shared with most agents

        #don't consider options that have already been tried (delete them from consideration)
        if len(self.tried_keys) > 0:
             for key in self.tried_keys:
                last_key_i = np.where(unique==key)
                unique = np.delete(unique, last_key_i)
                counts = np.delete(counts, last_key_i)

        action = self.keys[unique[np.argmax(counts)]]
        self.tried_keys.append(action)
        self.last_grid = copy.deepcopy(grid)

        return key_converter(action) #use this to index next key

    def predict_contingency(self, env):
        # Get env state
        grid, avail, agents, target, SELF = env.get_grid_state()
        print('agents: ', agents)
        print('self: ', SELF)
        set_trace()
        
        if self.last_action == None:
            self.last_grid = copy.deepcopy(grid)
            self.last_avail = copy.deepcopy(avail)
            self.last_agents = copy.deepcopy(agents)
            self.last_s = copy.deepcopy(SELF)
            self.last_action = key_converter(env.action_space.sample())
            print('first action: ', self.last_action)
            return self.last_action

        # get direction in which agents moved
        dirs = agents - self.last_agents
        #count number of directions = self.last_action

        # if more than one agent moved in the direction you chose
            # save those agents as ones as candidates, and choose action in different dimension. 
            # self is the only agent of these candidates that responded to the new instruction. 
        # Navigate self to the goal (using the old code?)
             
            

