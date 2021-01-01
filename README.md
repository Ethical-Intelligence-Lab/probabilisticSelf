# probabilisticSelf2

Testing what a self is for. Comparison of random, RL, theory-based agents, custom self class, and humans. 

Environment was adapted from: https://github.com/xinleipan/gym-gridworld
[How You Act Tells a Lot: Privacy-Leakage Attack on Deep Reinforcement Learning](https://arxiv.org/abs/1904.11082)

## Requirements

- Windows, OS X, or Linux 
- Python 3.5.2 (later versions might be fine)
- Tensorflow
- See `requirements.txt` for all other dependencies. 
- Gpu not necessary

##  Basic set up
(1) Clone the repo
        ```
        git clone https://github.com/juliandefreitas/probabilisticSelf2.git;
        cd probabilisticSelf2
        ```

(2) Create virtualenv in python 3.6--either from a conda environment or from your system (via python3 xyz)--and install requirements
        ```
        python -m venv env [make sure this is created using python 3.6]
        source env/bin/activate
        pip install -r requirements.txt
        ```

(3) Clone other relevant repos and rename them to avoid conflicts with official packages
        ```
        git clone https://github.com/openai/gym.git; #[2c50315 of master]
        mv gym gym_l #rename gym to gym_l, to avoid confusion with pip package;

        git clone https://github.com/hill-a/stable-baselines.git; #[259f278 of master]
        mv stable-baselines baselines_l 
        ```

(4) You may need to also implement the following tensorflow fixes.
        ```
        pip uninstall tensorflow-gpu;
        pip install tensorflow==1.15.0 --ignore-installed 
        ```
        #You may also run into a “Python is not installed as a framework” error, see: https://stackoverflow.com/questions/31373163/anaconda-runtime-error-python-is-not-installed-as-a-framework/41433353

##  Human player
        ```
        python main.py -p human -exp [your initials here]
        ```
    - You should see a window open. Make sure it is visible at the same time as your terminal. 
    - Enter keys in your terminal to move the agent: w = up, s = down, a = left, d = right. Hit 'enter' after each choice. Complete 100 levels. The script will exit upon completion.