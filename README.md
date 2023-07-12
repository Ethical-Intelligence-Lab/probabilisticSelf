# probabilisticSelf2

Testing what a self is for. Comparison of random, RL, theory-based agents, custom self class, and humans. 

Environment was adapted from: https://github.com/xinleipan/gym-gridworld
[How You Act Tells a Lot: Privacy-Leakage Attack on Deep Reinforcement Learning](https://arxiv.org/abs/1904.11082)

## Requirements for running RL agents (except Option-Critic)

- Windows, OS X, or Linux 
- Python 3.6 (later versions might be fine)
- Tensorflow
- See `requirements.txt` for all other dependencies. 
- Gpu not necessary

##  Basic set up
(1) Clone the repo

        ```
        git clone https://github.com/juliandefreitas/probabilisticSelf2.git;
        cd probabilisticSelf2
        ```

(2) Create virtualenv in python 3.6--either from a conda environment or from your system (via python3 xyz)

        ```
        python -m venv env [make sure this is created using python 3.6]
        source env/bin/activate
        ```

(3) Clone other relevant repos and rename them to avoid conflicts with official packages -- and install requirements

        ```
        git clone https://github.com/akaanug/stable-baselines.git
        mv stable-baselines baselines_l 
        pip install ./baselines_l
        pip install -r requirements.txt # Make sure you run this after installing baselines_l
        ```
(4) You may need to install OpenMPI, CMake, and zlib. For installation, please refer here: https://stable-baselines.readthedocs.io/en/master/guide/install.html

(5) You may also need to also implement the following tensorflow fixes.
        ```
        pip uninstall tensorflow-gpu;
        pip install tensorflow==1.15.0 --ignore-installed 
        ```
        #You may also run into a “Python is not installed as a framework” error, see: https://stackoverflow.com/questions/31373163/anaconda-runtime-error-python-is-not-installed-as-a-framework/41433353

##  Playing the script 

    - Basic command to play the script using default params (see params/default_params.py):
        ```
        python main.py 
        ```
    - To watch the game as it plays and toggle player and game type:
        ```
        python main.py -verbose 1 -player human -game_type contingency
        ```
    - For more parameters, see `params/param_dicts.py`

(6) Data
- Available in the project's OSF repository. Simply unzip the .zip folder in the project repo, such that the 'data' directory is visible. 


## Setup for Running Option-Critic:
- Create a Python 3.9 environment.
- Install the requirements in `requirements_hrl.txt`
- Install ray nightly from the wheel `pip install -U LINK_TO_WHEEL.whl`. You can find the links to install the wheel here: https://docs.ray.io/en/latest/ray-overview/installation.html
- Set sb to False in `gym_gridworld/gridworld_env.py`
- Run `main_option_critic.py` with the desired parameters.