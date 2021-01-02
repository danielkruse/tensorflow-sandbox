# Tensorflow 1D example

This is meant to be an example project for learning TensorFlow and understanding aspects of Deep Learning using modern code libraries

## Dependencies

Tested using Ubuntu 18.04 and Python 3.6.9

Check if this is still valid by checking [Installing Tensorflow using pip](https://www.tensorflow.org/install/pip)

This code uses Tensorflow >= 2.4

*Make sure to use* `python3` *in place of* `python` *if multiple installations of* `python` *are installed on your system*

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
python -m pip install --user matplotlib numpy pandas scipy tensorflow keras
python -m pip install --upgrade --user tensorflow
python -m pip install --upgrade --user keras
```

* `matplotlib` - Data plotting library intended to imitate data visualization used in MATLAB  
* `numpy` - Numerical Python Mathematics Library
* `pandas` - Python Data Analysis Library
* `scipy` - Scientific Computation Library based on NumPy useful for statistics and signal processing

## OpenAI Gym

Check steps at [OpenAI Gym Github](https://github.com/openai/gym) before continuing

Recommended steps are to install via steps

```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

### Installing custom gym

Check steps at [OpenAI Creating Environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

```bash
pip install -e custom-gym
```