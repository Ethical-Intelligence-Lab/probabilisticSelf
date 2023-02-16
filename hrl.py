"""Example of hierarchical training using the multi-agent API.
The example env is that of a "windy maze". The agent observes the current wind
direction and can either choose to stand still, or move in that direction.
You can try out the env directly with:
    $ python hierarchical_training.py --flat
A simple hierarchical formulation involves a high-level agent that issues goals
(i.e., go north / south / east / west), and a low-level agent that executes
these goals over a number of time-steps. This can be implemented as a
multi-agent environment with a top-level agent and low-level agents spawned
for each higher-level action. The lower level agent is rewarded for moving
in the right direction.
You can try this formulation with:
    $ python hierarchical_training.py  # gets ~100 rew after ~100k timesteps
Note that the hierarchical formulation actually converges slightly slower than
using --flat in this example.
"""

import argparse
from gymnasium.spaces import Discrete, Tuple
import logging
import os
from gym_gridworld.envs.gridworld_env import GridworldEnv, HierarchicalGridworldEnv

from ray.tune import register_env

from params.param_dicts import params

from gymnasium.wrappers import EnvCompatibility
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.windy_maze_env import WindyMazeEnv, HierarchicalWindyMazeEnv
from params.default_params import DefaultParams

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--flat", action="store_true")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--episodes-total", type=int, default=2000, help="Number of episodes to train."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
         "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

# Parameters for gridworld:
for k, v in params.items():
    parser.add_argument('--{}'.format(str(k)), type=type(k), default=v)

args = parser.parse_args()

logger = logging.getLogger(__name__)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    def_params = DefaultParams(player="ppo", baselines_v=-1)
    def_params.update_params(vars(args))
    P = def_params.params

    register_env("hier_env", lambda config: HierarchicalGridworldEnv(P))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.framework == "torch" else CustomModel
    )

    stop = {
        "episodes_total": args.episodes_total
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }


    env = EnvCompatibility(GridworldEnv(None))

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("low_level_"):
            return "low_level_policy"
        else:
            return "high_level_policy"


    config = (
        PPOConfig()
        .environment("hier_env")
        .framework(args.framework)
        .rollouts(num_rollout_workers=0)
        .training(entropy_coeff=0.01,
                  model={
                      "custom_model": "my_model",
                      "vf_share_layers": True,
                  }
                  )
        .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    env.observation_space,
                    Discrete(4),
                    PPOConfig.overrides(gamma=0.9),
                ),
                "low_level_policy": (
                    None,
                    Tuple([env.observation_space, Discrete(4)]),
                    env.action_space,
                    PPOConfig.overrides(gamma=0.0),
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    if args.no_tune:
        print("Running manual train loop without Ray Tune.")
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                    result["timesteps_total"] >= args.stop_timesteps
            ):
                break
        algo.stop()
    else:
        print("Training automatically with Ray Tune")
        results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
