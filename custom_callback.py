import os

from stable_baselines.common.callbacks import BaseCallback


# Callback to save the model every n steps and at the end
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, params, verbose=0, n=10000):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.params = params
        self.n = n

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        game_name_pf = "_game_shuffled/" if self.params['shuffle_keys'] else "_game/"

        path = 'saved_models/' + self.params['game_type'] + game_name_pf + self.params['player'] + '/' + \
                    "seed" + str(self.params['seed']) + "/lr" + str(self.params['learning_rate']) + "_gamma" + str(self.params['gamma']) + \
                    "_ls" + str(self.params['learning_starts']) + '_s' + \
                    str(int(self.params['shuffle_keys'])) + "_prio" + str(int(self.params['prioritized_replay'])) + "_" +\
                    str(int((self.num_timesteps/1000))) + "k/weights"



        if self.num_timesteps % self.n == 0:
            if not os.path.exists(path):
                os.makedirs(path)
            self.model.save(path)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Save for the last time before exiting
        game_name_pf = "_game_shuffled/" if self.params['shuffle_keys'] else "_game/"
        path = 'saved_models/' + self.params['game_type'] + game_name_pf + self.params['player'] + '/' + \
                    "seed" + str(self.params['seed']) + "/lr" + str(self.params['learning_rate']) + "_gamma" + str(
            self.params['gamma']) + \
                    "_ls" + str(self.params['learning_starts']) + '_s' + \
                    str(int(self.params['shuffle_keys'])) + "_prio" + str(
            int(self.params['prioritized_replay'])) + "_lastSave/weights"

        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save(path)
