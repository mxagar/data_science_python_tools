# Data Science Tools in Python

This project contains notebooks and notes related to the most important concepts and tools necessary for **machine learning** and **data science**.

I created most of the notebooks and notes while following several web tutorials and Udemy courses, such as:

- [Python for Data Sciene and Machine Learning Bootcamp (by José Marcial Portilla)](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
- [Complete Tensorflow 2 and Keras Deep Learning Bootcamp (by José Marcial Portilla)](https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/)
- [Python for Computer Vision with OpenCV and Deep Learning (by José Marcial Portilla)](https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/)
- [Practical AI with Python and Reinforcement Learning (by José Marcial Portilla)](https://www.udemy.com/course/practical-ai-with-python-and-reinforcement-learning/)
- [Machine Learning A-Z™: Hands-On Python & R In Data Science (by Kirill Eremenko & Hadelin de Ponteves)](https://www.udemy.com/course/machinelearning/)

Unfortunately, sometimes I have not found a repository to fork, so the attribution is done in this README.

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

For running most of the notebooks, I use the two environments (ds & tf), created as follows:
```bash
# All things related to ML without TF
conda create -n ds python=3.7
conda install jupyter numpy pandas matplotlib scipy sympy cython numba pytables jupyterlab
conda install scikit-learn scikit-image
conda install -c pytorch pytorch
conda install statsmodels
conda install seaborn
conda install pandas-datareader
pip3 install opencv-python
pip3/conda install ipympl
pip3 install plotly
pip3 install cufflinks
pip3 install chart-studio
pip3 install jupyterlab "ipywidgets>=7.5"
conda install -c conda-forge nodejs 
conda install -c conda-forge/label/gcc7 nodejs 
conda install -c conda-forge/label/cf201901 nodejs 
conda install -c conda-forge/label/cf202003 nodejs
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
# All things related to ML with TF
conda create -n tf tensorflow
conda install jupyter
conda install numpy pandas matplotlib scipy sympy cython numba pytables jupyterlab
conda install scikit-learn scikit-image
conda install -c pytorch pytorch
conda install statsmodels
conda install seaborn
conda install pandas-datareader
pip3 install opencv-python
conda install ipympl
```
