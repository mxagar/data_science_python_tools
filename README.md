# Data Science Tools in Python

This project contains notebooks and notes related to the most important concepts and tools necessary for **machine learning** and **data science**.

I started collecting most of the notebooks and notes while following several web tutorials and Udemy courses, such as:

- [Python for Data Sciene and Machine Learning Bootcamp (by José Marcial Portilla)](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
- [Complete Tensorflow 2 and Keras Deep Learning Bootcamp (by José Marcial Portilla)](https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/)
- [Python for Computer Vision with OpenCV and Deep Learning (by José Marcial Portilla)](https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/)
- [Practical AI with Python and Reinforcement Learning (by José Marcial Portilla)](https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/)
- [Machine Learning A-Z™: Hands-On Python & R In Data Science (by Kirill Eremenko & Hadelin de Ponteves)](https://www.udemy.com/course/machinelearning/)

Unfortunately, sometimes I have not found a repository to fork, so the attribution is done in this `README.md`.

The aforementioned courses are very practical, they don't focus so much on the theory; for that purpose, I used:

- "An Introduction to Statistical Learning with Applications in R", by James et al. A repository with python notebooks can be found in [https://github.com/JWarmenhoven/ISLR-python](https://github.com/JWarmenhoven/ISLR-python).
- "Reinforcement Learning" by Sutton & Barto.
- "Pattern Recognition and Machine Learning" by Bishop. A repository with python notebooks can be found in [https://github.com/ctgk/PRML](https://github.com/ctgk/PRML).

Note that in some cases I also just simply followed the documentation provided in the websites of the used packages.

Important related `howto` files (not public) of mine are (for my personal tracking):

- `~/Dropbox/Learning/PythonLab/python_manual.txt`
- `~/Dropbox/Documentation/howtos/sklearn_scipy_sympy_stat_guide.txt`
- `~/Dropbox/Documentation/howtos/keras_tensorflow_guide.txt`
- `~/Dropbox/Documentation/howtos/pybullet_openai_guide.txt`
- `~/Dropbox/Documentation/howtos/python_reinforcement_learning_openai.txt`

To run the notebooks locally, first, install an environment manager, e.g., [conda](https://docs.conda.io/en/latest/), create an environment and install the required dependencies:

```bash
# Create your env
conda create --name ds pip python=3.8
conda activate ds

# Install all necessary packages
# FIXME: Many packages can be removed
pip install -r requirements.txt
```

Then, you open the notebooks; if I were a beginner, I'd start sequentially.

Mikel Sagardia, 2018.  
No guarantees.
