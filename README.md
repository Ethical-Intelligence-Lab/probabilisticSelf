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
- (1) Clone the repo `git clone https://github.com/juliandefreitas/probabilisticSelf2.git`
- (2) Create a virtual environment 
        ```
        cd probabilisticSelf2;
        git checkout main;
        python3 -m venv env;
        source env/bin/activate
        ```
- (3) Install packages `pip install -r requirements.txt` 
- (4) Clone tfutils `git clone https://github.com/megsano/tfutils.git; cd tfutils; git checkout -t origin/interaction_utils; python3 setup.py install`

##  Human player
        ```
        python main.py -p human -exp [your initials here]
        ```
    - You should see a window open. Make sure it is visible at the same time as your terminal. 
    - Enter keys in your terminal to move the agent: w = up, s = down, a = left, d = right. Hit 'enter' after each choice. Complete 100 levels. The script will exit upon completion.