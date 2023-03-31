import itertools
import math
import random
from utils.keys import *
import copy


class Self_class():
    def __init__(self, seed):
        self.me = None
        self.seed = seed
        self.tried_keys = []
        self.keys = [0, 1, 2, 3]  # U, D, L, R
        self.action_ref = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.prev_move_dirs = [[], [], [], []]
        self.prev_actions = []
        self.agent_locs = []
        self.last_grid = []
        self.prev_action = None
        self.candidates = []
        self.action_counter = 0
        self.prev_candidates = []
        self.movements = []
        self.mode = 'self_discovery'
        self.prefer_vertical = False

        random.seed(self.seed)

    def get_candidates(self, cur_agents):
        self.candidates = []
        last_action = self.action_ref[self.prev_action]
        # Get direction in which agents moved
        for i, loc in enumerate(cur_agents):
            self.movements.append([cur_agents[i][0] - self.prev_agents[i][0],
                                   cur_agents[i][1] - self.prev_agents[i][1]
                                   ])

        # Consider as candidates only agents who moved in same direction as your action
        for i, move in enumerate(self.movements):
            if move == last_action:
                self.candidates.append(i)

    def navigate(self, target, SELF, env):
        dist_vertical = target[0] - SELF[0]
        dist_horiz = target[1] - SELF[1]

        def pick_move():
            if env.game_type == 'contingency_extended':
                if abs(dist_vertical) == 1 or abs(dist_horiz) == 1:

                    if abs(dist_vertical) == 1 and dist_horiz > 0:
                        # Do the same move to keep the distance of one horizontally, to avoid the mock self
                        return key_converter(3)
                    elif abs(dist_vertical) == 1 and dist_horiz < 0:
                        return key_converter(2)
                    elif abs(dist_horiz) == 1 and dist_vertical > 0:
                        return key_converter(1)
                    elif abs(dist_horiz) == 1 and dist_vertical < 0:
                        return key_converter(0)

            if self.prefer_vertical:
                if dist_vertical > 0:
                    next_move = key_converter(1)
                elif dist_vertical < 0:
                    next_move = key_converter(0)
                elif dist_horiz > 0:
                    if dist_horiz == 1 and dist_vertical == 0:
                        self.last_grid = []
                        self.tried_keys = []
                    next_move = key_converter(3)
                elif dist_horiz < 0:
                    if dist_horiz == -1 and dist_vertical == 0:
                        self.last_grid = []
                        self.tried_keys = []
                    next_move = (key_converter(2))
            else:
                if dist_horiz > 0:
                    next_move = (key_converter(3))
                elif dist_horiz < 0:
                    next_move = (key_converter(2))
                elif dist_vertical > 0:
                    if dist_vertical == 1 and dist_horiz == 0:
                        self.last_grid = []
                        self.tried_keys = []
                    next_move = key_converter(1)
                elif dist_vertical < 0:
                    if dist_vertical == -1 and dist_horiz == 0:
                        self.last_grid = []
                        self.tried_keys = []
                    next_move = key_converter(0)
            return next_move
        
        next_move = pick_move()
            
        # Avoid the non-self that is on the way
        if env.game_type == 'contingency_extended':
            if [SELF[0] + self.action_ref[next_move][0], SELF[1] + self.action_ref[next_move][1]] == env.mock_s.location:
                # Try the other move
                invalid_move = next_move
                while next_move != invalid_move:
                    print("Invalid move, trying")
                    next_move = pick_move()

                return next_move

            return next_move


        else:
            return next_move

    def predict(self, env):
        action = None
        if env.game_type == 'logic':
            action = self.predict_logic(env)
        if env.game_type in ['contingency', 'contingency_extended']:
            if env.shuffle_keys == False:
                action = self.predict_contingency(env)
            else:
                action = self.predict_shuffled(env)
        if env.game_type in ['change_agent', 'change_agent_extended_1', 'change_agent_extended_2']:
            action = self.predict_change_agent(env)
        return action

    def predict_logic(self, env):
        # Get state
        grid, avail, agents, target, non_self, SELF, mock_s = env.get_grid_state()
        around = [[], [], [], []]
        around_clear = [[], [], [], []]

        ''' (1) If any avatar is closer to the target, move that one '''
        if (len(self.last_grid) > 0):
            diff = grid - self.last_grid
            if (sum(diff.flatten() != 0)):
                print('navigating the self!')
                action = self.navigate(target, SELF, env)

                return action

        ''' (2) ...else, pick the option most likely to localize the self '''
        print('FINDING THE SELF!')

        # Get indeces of blocks around agents (regardless of their contents)
        for i, agent in enumerate(agents):
            around[i].append(grid[agent[0] - 1, agent[1]])
            around[i].append(grid[agent[0] + 1, agent[1]])
            around[i].append(grid[agent[0], agent[1] - 1])
            around[i].append(grid[agent[0], agent[1] + 1])
        around = np.array(around)
        zeros = np.transpose(np.nonzero(around == 0))  # which blocks have zeros
        unique, counts = np.unique(zeros[:, 1],
                                   return_counts=True)  # which available direction is shared with most agents

        # don't consider options that have already been tried (delete them from consideration)
        if len(self.tried_keys) > 0:
            for key in self.tried_keys:
                last_key_i = np.where(unique == key)
                unique = np.delete(unique, last_key_i)
                counts = np.delete(counts, last_key_i)

        action = self.keys[unique[np.argmax(counts)]]
        self.tried_keys.append(action)
        self.last_grid = copy.deepcopy(grid)

        print("LOGIC PREDICTED: ", action)
        self.prefer_vertical = True if random.randint(0, 1) == 1 else False

        return key_converter(action)  # use this to index next key

    def predict_contingency(self, env):
        self.action_counter += 1

        # Get env state
        grid, avail, agents, target, non_self, SELF, mock_s = env.get_grid_state()

        # Whenever environment resets, we start in self discovery mode
        if len(self.last_grid) == 0:
            self.mode = 'self_discovery'
            self.prev_action = None
            self.prev_candidates = []

        self.last_grid = copy.deepcopy(grid)

        # Get current sorted agents
        cur_agents = []
        cur_agents.extend(non_self)

        if mock_s.location != []:  # If mock self exists
            non_self.append(mock_s.location)

        cur_agents.append(SELF)
        self.movements = []
        self.candidates = []

        if self.mode == 'navigation':
            print("*** Navigating ***")
            action = self.navigate(target, SELF, env)
            return action

        # (1) First action.. 
        elif self.prev_action == None:
            print('*** Taking first action ***')
            self.prefer_vertical = True if random.randint(0, 1) == 1 else False
            self.prev_agents = copy.deepcopy(cur_agents)
            self.prev_action = 0  # move up (note: it's arbitrary which direction we move first)
            return self.prev_action

        # (2) Not the first action
        elif (self.prev_action != None):
            print('*** Getting self candidates ***')
            self.get_candidates(cur_agents)

            # (3) If only one agent moved in direction of keypress, navigate it to the reward 
            if (len(self.candidates) == 1):
                print('*** Found myself! Navigating to the reward ***')
                self.mode = 'navigation'
                self.prev_action = self.navigate(target, SELF, env)
                self.prev_candidates = copy.deepcopy(self.candidates)
                self.candidates = []
                return self.prev_action

            elif len(self.candidates) > 1:
                # (4) If > 1 agent moved in directon of keypress, take action in a different dimension 
                if len(self.prev_candidates) == 0:
                    print('*** Found > 1 self candidate. Taking another action ***')
                    self.prev_candidates = copy.deepcopy(self.candidates)
                    self.candidates = []
                    self.prev_agents = copy.deepcopy(cur_agents)
                    self.prev_action = 2  # move left (different dimension from first move)
                    return self.prev_action  # go up

                # (5) Otherwise, the self is the only agent among the candidates that moved in the direction of the keypress 
                elif len(self.prev_candidates) > 0:
                    print('*** Eliminating candidates. ***')
                    self.get_candidates(cur_agents)
                    candidate = set(self.candidates).intersection(self.prev_candidates)
                    final_candidate = candidate.pop()
                    if ('extended' in env.game_type and final_candidate == 4) or ('extended' not in env.game_type and final_candidate == 3):
                        self.prev_action = self.navigate(target, SELF, env)
                        self.mode = 'navigation'
                        return self.prev_action

    # Go back to self discovery if the agent has changed (moved into unexpected position)
    def check_if_agent_is_correct(self, action):
        self.prev_candidates = []

        new_loc = [self.agent_locs[self.action_counter - 2][0] + action[0],
                   self.agent_locs[self.action_counter - 2][1] + action[1]]

        print("guessed self: ", new_loc)
        print("self: ", list(self.agent_locs[self.action_counter - 1]))
        print("action: ", action)

        if list(self.agent_locs[self.action_counter - 1]) != new_loc:
            print("Unexpected position: going back to self discovery")
            self.mode = "self_discovery"

    # Get which direction to move based on agents' locations.
    # e.g if > 2 of them are above and at the leftside the goal, then the proper move is to move down or right
    def get_direction(self):
        boundary = [10, 10]
        counts = [0] * 4  # U D L R
        actions = []

        for agent in self.agent_locs:
            if agent[0] > 10:
                counts[2] = counts[2] + 1  # increment left
            elif agent[0] < 10:
                counts[3] = counts[3] + 1  # increment right

            if agent[0] > 10:
                counts[0] = counts[0] + 1  # increment up
            elif agent[0] < 10:
                counts[1] = counts[1] + 1  # increment down

        if counts[0] > counts[1]:
            actions.append(0)  # UP
        elif counts[1] > counts[0]:
            actions.append(1)  # DOWN
        else:
            actions.append(random.randint(0, 1))

        if counts[2] > counts[3]:
            actions.append(2)  # LEFT
        elif counts[3] > counts[2]:
            actions.append(3)  # RIGHT
        else:
            actions.append(random.randint(2, 3))

        return actions[random.randint(0, len(actions) - 1)]

    def predict_change_agent(self, env):
        print("*-*-*-*---- PREDICTING ----*-*-*-*")
        self.action_counter += 1

        # Get env state
        grid, avail, agents, target, non_self, SELF, mock_s = env.get_grid_state()
        print(env.game_type)
        self.agent_locs.append(SELF)

        # Whenever environment resets, we start in self discovery mode
        if len(self.last_grid) == 0:
            self.mode = 'self_discovery'
            self.prev_action = None
            self.prev_candidates = []

        self.last_grid = copy.deepcopy(grid)

        # Get current sorted agents
        cur_agents = []
        cur_agents.extend(non_self)
        cur_agents.append(SELF)

        self.movements = []
        self.candidates = []

        print("SELF: ", SELF)

        if self.mode == 'navigation':
            print("*** Navigating ***")
            self.check_if_agent_is_correct(self.action_ref[self.prev_action])
            action = self.navigate(target, SELF, env)
            return action

        # (1) First action..
        elif self.prev_action == None:
            print('*** Taking first action ***')
            self.prefer_vertical = True if random.randint(0, 1) == 1 else False
            self.prev_agents = copy.deepcopy(cur_agents)
            self.prev_action = key_converter(random.randint(0, 3))
            return self.prev_action

        # (2) Not the first action
        elif (self.prev_action != None):
            print('*** Getting self candidates ***')
            self.get_candidates(cur_agents)

            # (3) If only one agent moved in direction of keypress, navigate it to the reward
            if (len(self.candidates) == 1):
                if ("extended" in env.game_type and self.candidates[0] == 4) or ("extended" not in env.game_type and self.candidates[0] == 3):
                    print('*** Found myself! Navigating to the reward ***')
                    self.mode = 'navigation'
                    self.prev_action = self.navigate(target, SELF, env)
                    self.prev_candidates = copy.deepcopy(self.candidates)
                    self.candidates = []
                    return self.prev_action
                else:
                    print("Eliminated candidate is not self.")
                    self.prev_candidates = []
                    self.prev_agents = copy.deepcopy(cur_agents)
                    self.prev_action = key_converter(self.get_direction())
                    return self.prev_action

            elif len(self.candidates) > 1:
                # (4) If > 1 agent moved in direction of keypress, take action in a different dimension
                if len(self.prev_candidates) == 0:
                    print('*** Found > 1 self candidate. Taking another action ***')
                    self.prev_candidates = copy.deepcopy(self.candidates)
                    self.candidates = []
                    self.prev_agents = copy.deepcopy(cur_agents)
                    self.prev_action = key_converter(self.get_direction())
                    return self.prev_action

                #  More than one candidates: Eliminate
                elif len(self.prev_candidates) > 0:
                    print('*** Eliminating candidates. ***')

                    self.get_candidates(cur_agents)

                    candidate = set(self.candidates).intersection(self.prev_candidates)

                    print("Eliminated candidates: ", candidate)

                    if len(candidate) == 0:
                        print("No matching candidates")
                        self.prev_agents = copy.deepcopy(cur_agents)
                        self.prev_action = key_converter(self.get_direction())
                        return self.prev_action
                    elif len(candidate) > 1:
                        print('*** Found > 1 self candidate. Taking another action ***')
                        self.prev_candidates = copy.deepcopy(self.candidates)
                        self.candidates = []
                        self.prev_agents = copy.deepcopy(cur_agents)
                        self.prev_action = key_converter(self.get_direction())
                        return self.prev_action

                    # If one candidate:
                    final_candidate = candidate.pop()
                    if ("extended" in env.game_type and final_candidate == 4) or ("extended" not in env.game_type and final_candidate == 3):
                        print("Found self, going into navigation.")
                        self.prev_candidates = []
                        self.prev_action = self.navigate(target, SELF, env)
                        self.mode = 'navigation'
                        return self.prev_action
                    else:
                        print("Eliminated candidate is not self.")
                        self.prev_candidates = []
                        self.prev_agents = copy.deepcopy(cur_agents)
                        self.prev_action = key_converter(self.get_direction())
                        return self.prev_action

            else:  # No candidates.
                print("No candidates. Keep moving")
                self.prev_agents = copy.deepcopy(cur_agents)
                self.prev_action = key_converter(self.get_direction())
                return self.prev_action

    def predict_shuffled(self, env):
        self.action_counter += 1
        print('action counter: ', self.action_counter)

        # Get env state
        grid, avail, agents, target, non_self, SELF, mock_s = env.get_grid_state()

        # Whenever environment resets, we set self.mode to 'self discovery'
        if len(self.last_grid) == 0:
            self.mode = 'self_discovery'
            self.prev_action = None
            self.prev_candidates = []
            self.prev_move_dirs = [[], [], [], []]
            self.action_counter = 1
        self.last_grid = copy.deepcopy(grid)

        # Get current sorted agents
        cur_agents = []
        cur_agents.extend(non_self)
        cur_agents.append(SELF)
        self.movements = []
        self.candidates = []

        # (1) First action.. 
        if (self.mode == 'self_discovery') & (self.prev_action == None):
            print('*** Taking first action ***')
            self.prefer_vertical = True if random.randint(0, 1) == 1 else False
            self.prev_actions = []
            self.prev_agents = copy.deepcopy(cur_agents)
            self.prev_action = 0  # (note: it's arbitrary which direction we move first)
            self.prev_actions.append(self.prev_action)
            return self.prev_action

        # (2) Subsequent actions
        elif (self.mode == 'self_discovery') & (self.prev_action != None):
            print('*** Taking action #' + str(self.action_counter))

            # Get direction in which agents moved
            self.update_steps(cur_agents)
            self.prev_agents = copy.deepcopy(cur_agents)

            if self.action_counter == 2:
                self.prev_action = 1
                self.prev_actions.append(self.prev_action)
                return self.prev_action

            # if you took more than 2 actions, see if the actions spanned different dimensions
            elif self.action_counter == 3:
                for i, entry in enumerate(self.dir_summary):
                    if (1 in entry) & (0 in entry) & (i == 3):
                        self.mode = 'navigation'
                        print('*** Navigating ***')
                        self.prev_action = self.navigate_shuffled(target, SELF)

                if self.mode == 'self_discovery':
                    print('keep eliminating')
                    self.prev_action = 2

            # after taking 3 actions, we must have sampled two dimensions
            elif self.action_counter == 4:
                assert (1 in self.dir_summary[3]) & (0 in self.dir_summary[3])
                self.mode = 'navigation'
                self.prev_action = self.navigate_shuffled(target, SELF)

            # self.update_steps(cur_agents)
            self.prev_agents = copy.deepcopy(cur_agents)
            self.prev_actions.append(self.prev_action)
            return self.prev_action

        elif self.mode == 'navigation':
            self.update_steps(cur_agents)
            self.prev_agents = copy.deepcopy(cur_agents)
            self.prev_action = self.navigate_shuffled(target, SELF)
            self.prev_actions.append(self.prev_action)
            return self.prev_action

    def update_steps(self, cur_agents):
        # Save directions in which agents moved
        for i, loc in enumerate(cur_agents):
            self.prev_move_dirs[i].append([cur_agents[i][0] - self.prev_agents[i][0],
                                           cur_agents[i][1] - self.prev_agents[i][1]
                                           ])

        print('prev move dirs: ', self.prev_move_dirs[3])
        print('prev actions: ', self.prev_actions)

        # Summarize the dimensions sampled by these actions
        if len(self.prev_actions) >= 2:
            self.dir_summary = [[], [], [], []]
            for i, direction in enumerate(self.prev_move_dirs):
                for k in range(0, self.action_counter - 1):
                    indeces = [i for i, e in enumerate(direction[k]) if e != 0]
                    if (len(indeces) > 0):
                        self.dir_summary[i].append(indeces[0])
                    elif (len(indeces) == 0):
                        self.dir_summary[i].append(3)

    def navigate_shuffled(self, target, SELF):
        print('navigating with shuffled keys')

        self.shuffle_key_dict = {}
        for i, action in enumerate(self.prev_actions):
            self.shuffle_key_dict[str(self.prev_move_dirs[3][i])] = self.prev_actions[i]

        print('prev move dirs: ', self.prev_move_dirs[3])
        print('direction summary: ', self.prev_actions)
        print('shuffled key dict: ', self.shuffle_key_dict)

        dist_vertical = target[0] - SELF[0]
        dist_horiz = target[1] - SELF[1]

        if self.prefer_vertical:
            if dist_vertical > 0:
                key = 1
            elif dist_vertical < 0:
                key = 0
            elif dist_horiz > 0:
                key = 3
                if dist_horiz == 1:
                    self.last_grid = []
            elif dist_horiz < 0:
                key = 2
                if dist_horiz == -1:
                    self.last_grid = []
        else:
            if dist_horiz > 0:
                key = 3
            elif dist_horiz < 0:
                key = 2
            elif dist_vertical > 0:
                key = 1
                if dist_vertical == 1:
                    self.last_grid = []
            elif dist_vertical < 0:
                key = 0
                if dist_vertical == -1:
                    self.last_grid = []

        print('dist vertical: ', dist_vertical)
        print('dist horizontal: ', dist_horiz)
        print('key: ', key)
        print('key code: ', self.action_ref[key])

        if str(self.action_ref[key]) in self.shuffle_key_dict:
            cur_key = self.shuffle_key_dict[str(self.action_ref[key])]
        else:
            print('key not in dictionary!')
            cur_key = max(self.prev_actions) + 1
        return key_converter(cur_key)
