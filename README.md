# Tensorflow 1D example

This is meant to be an example project for learning TensorFlow and understanding aspects of Deep Learning using modern code libraries

## Usage

### Supervised Learning

#### Tensorflow MNIST quickstart tutorial

This code is based on [Tensorflow Quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner) and has additional comments to help with understanding

While in project folder

```bash
cd supervised_learning
python tensorflow_mnist_quickstart.py
```

#### Tensorflow 1D-NN test script

This code is intended for visualizing the evolution of training fully connected Neural Networks over epochs with arbitrary 1D functions.  Trains a Neural Network against one of several predefined functions and then displays the evolution of training.

While in project folder

```bash
cd supervised_learning
python tensorflow_1dnn_test.py [-h] [-e EPOCHS] [-b BATCH] FUNCTION
```

* `-h`, `--help` show help message and exit
* `FUNCTION` name of function to train against.  Available functions are
  * `exponential`
  * `piecewise`
  * `quadratic`
  * `segmentation`
  * `sinusoid`
* `--epochs [-e] EPOCHS` epochs to train over (defaults 1000)
* `--batch [-b] BATCH` batch size (defaults 16)

### Reinforcement Learning

This code is intended for visualizing the results of Reinforcement Learning.  

#### Tensorflow DDPG inverse pendulum example

Example code from [OpenAI Spinning Up Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) this shows an implementation of running `DDPG` on the `Pendulum-v0` gym from `OpenAI`.

```bash
cd reinforcement_learning
python tensorflow_ddpg_pendulum.py
```

#### Tensorflow DDPG custom implementation

A custom implementation of [Deep Determinstic Policy Gradients](https://arxiv.org/abs/1509.02971) that will train (with `--train`) the actor and critic models and save the results to a folder `saved_models`.  Display can be called separately from training with these saved models using the `--display` flag.

```bash
cd reinforcement_learning
python tensorflow_ddpg_test.py [-h] [-t] [-d] [-e EPISODES] [--render_gym] GYM
```

* `GYM` name of gym to train against. Known gyms:
  * `custom_gym:custom-1d-gym-v0`
  * `custom_gym:custom-2d-gym-v0`
  * `custom_gym:custom-2d-obstacles-gym-v0`
  * `Pendulum-v0`
* `--help [-h]` show this help message and exit
* `--train [-t]` flag to train gym in DDPG
* `--display [-d]` flag to display most recent results from training
* `--episodes [-e] EPISODES` number of episodes to train over (default 100)
* `--render_gym` flag to render gym during training

## Dependencies

Tested using

* Ubuntu 18.04 with Python 3.6.9
* Windows 10 with Python 3.7.9

This code uses Tensorflow >= 2.4

Validate the following steps by checking [Installing Tensorflow using pip](https://www.tensorflow.org/install/pip)

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

### Installing CUDA GPU optimization

Note that TensorFlow 2.4 supports CUDA 11 according to [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu) which requires nVidia drivers 450.x or higher according to [nVidia CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

#### Ubuntu 18.04

Refer to [Install CUDA with APT](https://www.tensorflow.org/install/gpu#install_cuda_with_apt) for full instructions from TensorFlow

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-450
# Reboot. Check that GPUs are visible using the command: nvidia-smi

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0


# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0
```

#### Windows 10

Please refer to [GPU Windows setup](https://www.tensorflow.org/install/gpu#windows_setup) from TensorFlow and the [CUDA® install guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/). **TensorFlow will not load without the cuDNN64_8.dll file.**

Add the CUDA®, CUPTI, and cuDNN installation directories to the %PATH% environmental variable. For example, if the CUDA® Toolkit is installed to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0` and cuDNN to `C:\tools\cuda`, update your `%PATH%` to match

```cmd
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

## OpenAI Gym

Validate following steps at [OpenAI Gym Github](https://github.com/openai/gym) before continuing

Recommended steps are to install via steps (in external installation folder)

```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

### Installing custom gyms

Validate following steps at [OpenAI Creating Environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

While in project folder

```bash
cd reinforcement_learning
pip install -e custom-gym
```
