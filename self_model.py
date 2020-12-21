from pdb import set_trace
import numpy as np
from utils.keys import *
import copy

class Self_class():
    def __init__(self):
        self.me = None
        self.tried_keys = []
        self.keys = [1, 2, 3, 4] #U, D, L, R
        self.last_grid = []
    
    def predict(self, env):
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
                    return key_converter(2)
                elif dist_vertical < 0:
                    return key_converter(1)
                elif dist_horiz > 0:
                    if dist_horiz == 1:
                        self.last_grid = []
                        self.tried_keys = []
                    return (key_converter(4))
                elif dist_horiz < 0:
                    if dist_horiz == -1:
                        self.last_grid = []
                        self.tried_keys = []
                    return (key_converter(3))

        # (2) ...else, pick the option most likely to localize the self
        print('FINDING THE SELF!')

        # Get indeces of blocks around agents
        for i, agent in enumerate(agents):
            around[i].append(grid[agent[0]-1, agent[1]])
            around[i].append(grid[agent[0]+1, agent[1]])
            around[i].append(grid[agent[0], agent[1]-1])
            around[i].append(grid[agent[0], agent[1]+1])

        around = np.array(around) #options around each agent
        zeros = np.transpose(np.nonzero(around == 0)) #zeros around each agent
        unique, counts = np.unique(zeros[:,1], return_counts=True) #which direction is shared with most agent

        #dont' consider options that have already been tried
        if len(self.tried_keys) > 0:
             for key in self.tried_keys:
                last_key_i = np.where(unique==key -1)
                unique = np.delete(unique, last_key_i)
                counts = np.delete(counts, last_key_i)

        action = self.keys[unique[np.argmax(counts)]]
        self.tried_keys.append(action)
        self.last_grid = copy.deepcopy(grid)

        return key_converter(action) #use this to index next key
        
