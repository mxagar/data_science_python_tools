# Reinforcement Learning Notebooks

I made these notebooks following the Udemy course by José Portilla **Practical AI with Python and Reinforcement Learning**:

[https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/](https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/)

In addition to the notebooks in here, this course reviews other introductory concepts not explained here:
- Numpy
- Matplotlib
- Machine Learning concepts
- Pandas
- Scikit-Learn
- Keras: ANNs and CNNs

**Overview**:
1. Introduction and Setup
2. Reinforcement Learning Concepts
3. OpenAI Gym Overview
4. Classical Q-Learning
5. Deep Q-Learning
6. Deep Q-Learning on Images
7. Creating Custom OpenAI Gym Environments

## 1. Introduction and Setup

```bash
conda env list
conda activate tf
pip install jupyter numpy matplotlib 
# Install OpenAI Gym base library
pip install gym
# By default, only the classic control family is installed
# We need to install specific environment families manually
# Note that I had issues installing Box2D and MuJoCo
pip install 'gym[atari,accept-rom-license]'
pip install pygame
# We can also do in a Jupyter notebook: `!pip install gym`
# And we can also install stuff using Anaconda
conda install lxml pandas pillow scikit-learn seaborn tensorflow
```

Introductory sections, not covered here:
- Numpy: `~/Dropbox/Learning/PythonLab/python_manual.txt`
- Matplotlib: `~/Dropbox/Learning/PythonLab/python_manual.txt`
- Machine Learning concepts: `~/Dropbox/Documentation/howtos/sklearn_scipy_sympy_stat_guide.txt`
- Pandas: `~/Dropbox/Learning/PythonLab/python_manual.txt`
- Scikit-Learn: `~/Dropbox/Documentation/howtos/sklearn_scipy_sympy_stat_guide.txt`
- Keras: ANNs and CNNs: 
    - `~/Dropbox/Documentation/howtos/keras_tensorflow_guide.txt`
    - `~/Dropbox/Learning/PythonLab/udemy_cv_course/open_cv_python_manual.txt`

In particular, the guide on Keras collects all important notebook I have on the topic to date:
`~/Dropbox/Documentation/howtos/keras_tensorflow_guide.txt`.

## 2. Reinforcement Learning (RL) Concepts

Where does RL lie in the Machine Learning landscape?
- Supervised learning requires labelled data.
- Unsupervised learning has unlabelled data.
- Reinforcement learning does not use historical data, instead, RL uses rewarded repetition to learn a desired behavior on an environment. In other words: we don't have a dataset, but we act in an environment and see what happens.

Important elements:
- **Agent**: AI that can observe and interact with the environment performing actions; example: robot
    - Observations are often partial.
- **Environment**: the scenario where the game develops with its rules (physical and others); example: maze
    - The environment can change through the actions of the agent.
- **State** vs **Observation**: the state is the complete description of the environment, without hidden data; usually, what we get are observations, though, which are partial views of the state, with hidden or undiscovered pieces of information; note that both are often mixed of used interchangedly.
- **Reward**: we have a goal state in mind, which we would like to achieve. Afte r executing an action, we make observations and estimate the state; depending on how far we are from the goal state (or our improvement towards it, I understand), we assign a better (positive) or worse (negative) reward.
- **Policy**: a set of rules of the AI agent to decide what to do next depending on the observations and the reward. Note that often agent and policy are mixed or used interchangedly.
    - The policy is updated to maximize the reward in the direction of the expected goal state.

Typical example: Cart pole (in Spanish, *péndulo invertido*): The goal is to maintain the cart pole upright by moving the cart left/right; we get an observation of the pole's angle after our moving action and a reward is accordingly assigned depending on its angle.

### Markov Decision Process in a Gridworld and the Bellman Equation

I wrote this section after reading Wikipedia articles and watching the following video:

[A friendly introduction to deep reinforcement learning, Q-networks and policy gradients, by Luis Serrano](https://www.youtube.com/watch?v=SgC6AZss478).

The pictures in this section are taken from that video.

The goal of the section is to explain the Bellman equation, which is probably the most important equation in RL, since it defines how the transitions in the state space are defined so that the goal can be reached.

Let's consider a 6x6 **gridworld** in which an agent can move in 4 directions with one step at a time (i.e., 4 **actions**: up/down & left/right), except when boundaries or obstacles are hit; each possible cell is a **state** and can have a **value** assigned to it.

![Gridworld (by Luis Serrano)](pics/gridworld.png)

In that world there are three **terminal states** or cells in which the game ends; we would like to end in the state with the highest value. The process to achieve that is called a **Markov Decision Process**, which consists in executing actions available to the current state that cause transitions to states that are closer to the goal.

The **Bellman equation** is a recursive function which, if called many times, discovers for the complete world
- the **value function** $V()$ that assigns value to any state/cell in the environment
- and the **action policy** $a$ of each cell/state, i.e., the optimum action to take to reach the goal.

Let's say we go from one state to a next with a single action $a$: `s -> s'`; we assume:
- The *value* of each state is given by $V(s)$
- The action $a$ we take is the *optimal* one, i.e., we are trying to maximize the value
- Taking that action has a *reward* (watch out: cost due to action) relative to the state we are in: $R(s,a)$.

Then, the Bellman equation is

$V(s) = \max_{a}\{R(s,a) + \gamma V(s')\}$, where

$gamma$ is a **discount factor** $0 < \gamma < 1$, for instance $\gamma = 0.9$; for simplicity, we can consider it to be $\gamma = 1$.

Following the Bellman equation, we can deduce tha value of a cell/state by observing its neighbors; note that while we go from `s` to `s'` during navigation, for value assigning, we do the oposite: `s' -> s`. Thus, according to the Bellman equation, the value of `s` is the maximum possible value of the next neighbors ($V(s')$) plus the reward ($R(s,a)$), which is negative.

![Value propagation with the Bellman equation (by Luis Serrano)](pics/propagation.png)

With the Bellman equation we propagate the values of the cells/states from the known ones to the unknowns decreasing with the reward $R(s,a)$ at each step, as if it were a *distance field*.
Note that if we start propagating from different terminal states, inconsistencies might appear in the distance field; however, if we continuously re-run the Bellman equation in the map, the values will converge to a consistent function.
The best policy will be then to take the action that leads to the neighbor cell with the largest value.
Note that **the Bellman equation is usually solved by randomly visiting states/cells: if neighbor values are known, the value of the visited cell/state is updated, until enough states are known and converge**.
That is so because the state space might be very large to sweep and propagate all values.

![Bellman equation (by Luis Serrano)](pics/bellman.png)

The discount factor models tha fact that a state value in future steps will loose its value (because we are less sure); in our case, it can be considered $\gamma = 1$ for simplicity.

### Deterministic vs. Stochastic Policies

A **deterministic policy** maps a state to an action.

$a \leftarrow \mu(s)$

A **stochastic policy** assigns probabilities to the set of possible action available for the state; that way, we choose the one with the highest probability, but we are open to take other actions, too!

$\pi(a_i|s)$: $\pi(a_1|s)$, $\pi(a_2|s)$, $\pi(a_2|s)$, $\pi(a_4|s)$

That is also known as **exploitation** (one deterministic action) vs **exploration** (a set of possible actions to try).

![Deterministic vs. stochastic policies (by Luis Serrano)](pics/stochastic_vs_deterministic.png)

### Neural Netowks for Value & Policy Prediction

Since computing the **state values** $V(s)$ and the **action policies** (i.e., which actions to take in each state) might be very expensive for large worlds, one can train neural networks that predict them given past history; that is the key idea behind Deep Reinforcement Learning.

**Value networks** get the coordinates $(x,y)$ of a state/cell $s$ and output its value $V(s)$. The underlying assumption is that close states will have close values. They are trained wandering the environment and forcing the network to yield values compliant with the Bellman equation. This works in practice a s follows:
- We select a cell/state $s$
- We predict with the network its value and the value of its neighbors
- We compute the value of the cell according to the Bellman equation
- We have now a datapoint: the coordinates and the Bellman value
- We compute the error as the difference between the initial guessed value and the Bellman value with the neighbor values
- We apply backpropagation
- If these steps are repeated enough times, the state value predictions converge!

**Policy networks** get the coordinates $(x,y)$ of a state/cell $s$ and output the probability for each of the possible actions $\pi(a_i|s)$, in other words, the stochastic policy. The underlying assumption is that close states will have close policies. They are trained wandering the environment. They are trained as follows:
- We get a path from the network: we predict the policies for the cells and follow the path with the highest probability until we end in a terminal cell.
- We assign **gain** values to each cell backwards starting from the terminal cell: gains are assigned according to Bellman, that is, we basically decrease the final cell value at each step backwards with the reward.
- Our sample datapoints consist of: coordinates, action taken and gain.
- Since we achieved the target, we force the network to increase the action probabilities that yielded it, but with a trick: The gain is multiplied to the weight update during network optimization. That way, only high gains are really reinforced.

## 3. OpenAI Overview

### Brief History of OpenAI

2015: Elon Musk and Sam Altman announced the formation of OpenAI.
Sam Altman is the current CEO.
It started as a non-profit: benefits for humanity sought, goal of reducing the risk of potential harm by AI. They were concerned by AI algorithms being researched only on private companies.

OpenAI Gym is the library they published in 2016: it can be used to set up environments for reinforcement learning.

2018: GPT is published by OpenAI: Generative Pre-Trained algorithm able to produce meaningful/conversational text taking into account the context; the model was trained with a large corpus of data (e.g., the Wikipedia).

2018: Dactyl: a shadow hand trained only in simulations; the robotic hand was able to manipulate a cube after learning everything in the simulation.

2018: Musk left the OpenAI board due to the fact that Tesla was developing also AI algorithms.

2019: OpenAI becomes for-profit, but with a profit cap (100x).

June 2019: GPT-2 is announced, but no code/model provided - the reason according to OpenAI: because bad actors could misuse their model to produce fake new or related; they were criticised for that. Later in Vovember 2019 the model was completely published. It is huge: 1.5 billion parameters.

2020: GPT-3 is announced: 175 billion parameters, the largest language model ever created; Microsoft has the exclusive license.

2021: DALL-E paper is published by OpenAI: very nice Text-to image model. The model is not public.

### OpenAI Gym Documentation

The documentation page of OpenAI does not have extensive information:

[OpenAI Documentation](https://gym.openai.com)

The most important part is the one related to the environments; when we select one, we need to read the source to understand what they're about. They have also a link to the original paper where they were defined/suggested. The environments are classified as follows:
- Classical control: usually, we start here; the typical Cart-Pole and Mountain-Car scenarios are here.
- Atari: images of Atari games. We have 2 versions for some games: the one with RAM contains relevant information on some object positions, etc.; the other has only images and a CNN should be applied to understand the object poses.
- Box2D: 2D physics engine.
- MuJoCo: 3D physics engine, often the famous manequins that learn to walk.
- Algorithms: algorithms that learn to sort sequences, etc.
- Robotics: the Dactyl is here.
- Third party envs (link is broken at the moment).

### Gym Environments: Overview - `01_OpenAI_Gym_Overview.ipynb`

See notebook

`01_OpenAI_Gym_Overview.ipynb`

This notebook starts exploring the OpenAI Gym library with the following games:

1. [Atari / Breakout](https://gym.openai.com/envs/Breakout-ram-v0/)
2. [Classic control / Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)

Note 1: We have two game versions:
- `RAM` version: ball coordinates and paddle location are returned, not images; useful for simple environments.
- Standard: a history of images is returned; CNNs are required.

Note 2: if we use Jupyter notebooks, sometimes we need to restart the kernel; an alternative is to use python scripts.

Typical methods we need to know:
- `reset()`
- `step(action)`: e.g., `4: left`, etc.
- `render()`
    - `render("human")`: images rendered, for human beings
    - `render("rgb_array")`: numpy RGB array; for computers or visualizing with matplotlib

#### 1. Atari Breakout Game

Popular Atari game in which we m ove paddle so that we hit a ball that collides agains a rainbow ball; collisions remove rainbow blocks (goal). If we miss hitting the ball, it falls down (avoid).

Note that an extra environment needs to be installed, along with pygame:

```bash
pip install 'gym[atari,accept-rom-license]'
pip install pygame
```

This section shows how to:

- Create simulation loops
- Render images or numpy arrays
- Access actions and execute them

Summary of most important lines

```python
import gym
# For plotting
import matplotlib.pyplot as plt
# For slowing down the game
import time

# Select a game/environment from
# gym.openai.com
# If we go to Atari/Breakout-v0 we can see the source code
# We could use the source code file or let gym grab it
# as follows below.
# HOWEVER: Always have a look at the code!

# The string of the name is the title of the game
env_name = 'Breakout-v0'
#env_name = 'Breakout-ram-v0'

# The source code is grabbed
env = gym.make(env_name)

# We can interact/play from some Atari games
# and classic control envs.
# We need to have pygame installed: pip install pygame
from gym.utils import play

# We pass the env and zoom the window 2-3x
# Keys:
# - space: launch ball
# - a: move left
# - d: move right
# When we close the window, we might need to restart the kernel
play.play(env,zoom=3)

# Window will be opened and game rendered step-wise (but very fast)
# Nothing happens for now
# because no actions are commanded
for steps in range(2000):
    env.render(mode='human')

# Close env/game window
env.close()

# Now we render is as a numpy array/image
array = env.render(mode='rgb_array')

%matplotlib inline
plt.imshow(array)

# Action Space: how many actions can we execute?
env.action_space
env.action_space.n

# Let's render the game executing random actions
# A windw opens and displays the random game
# First, we need to always reset it to the initial state
_ = env.reset()
for step in range(200):
    env.render("human")
    # Random action (int), chosen uniformly
    # look at github/code the meaning of actions
    random_action = env.action_space.sample()
    # We performa steo passing the action
    # and we get 4 objects:
    # - observation
    # - reward given
    # - whether the game is finished
    # - game specific info
    observation, reward, done, info = env.step(random_action)
    print(f'Reward: {reward}')
    print(f'Done: {done}')
    print(f'Info: {info}')
    if done: #eg, if we run out of lives
        break
    # I we want to visualize it, we need to slow it down
    time.sleep(0.1)
env.close()
```

#### 2. Mountain Car

A very famous testbed published by Moore in 1990: "A car is on a one-dimensional track, positioned between two mountains. The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum."

Note that:
- There is gravity
- We have a single position variable in X axis; 0 appears to be the valley, 0.5 the flag
- The goal is to directly land the falg, no less, no more!

Links:
- [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)
- [Github link](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py)

```python
import gym
import time

env_name = 'MountainCar-v0'
env = gym.make(env_name)

# Look in the code to understand the meaning of the actions
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
# 0: accelerate to the left
# 1: stay put
# 2: accelerate to the right
env.action_space

# Note that this agent with the defined policy is able to climb
# the mountain, but we are not successful on landing exactly on the flag.
# Additionally, no rewards are used, i.e., this is not RL
# Instead, we see how we can interact with the environment
def simple_agent(observation):
    # Observation
    position, velocity = observation
    # When to go right
    if position > -0.1:
        action = 2
    # When to go left
    elif velocity < 0 and position < -0.2:
        action = 0
    # When to do nothing
    else:
        action = 1
    return action

    env.seed(42)
observation = env.reset()

for step in range(600):
    env.render(mode="human")
    action = simple_agent(observation)
    observation, reward, done, info = env.step(action)
    time.sleep(0.001)
env.close()

```