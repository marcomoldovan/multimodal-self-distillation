*This repository contains the code of two distinct research projects which are closely related and share much of the same codebase. The second project is and extension to the multimodal domain of the first one.*

<div align="center">

# General Representation Learning through Latent Space Augmentation and Prediction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

We want to generalize the self-distillation learning paradigm so that it applied to any kind of unimodal or fused multimodal data without the need of modality-specific augmentation or masking strategies. Instead we embed the input data into a universal input array and apply a single masking strategy in the latent space instead of the data space. We test this genealized apporach on a multitude of datasets containing text, images, audio and video data.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/marcomoldovan/cross-modal-speech-segment-retrieval
cd cross-modal-speech-segment-retrieval

# install the correct python version
sudo apt-get install python3.10 # Linux, Python 3.7 or higher
brew install python@3.10 #MacOS, Python 3.7 or higher
choco install python --version=3.9 # Windows, Python 3.7-3.9

# create python virtual environment and activate it
python3 -m venv myenv
source myenv/bin/activate

# if you have several version of python you can create a virtual environment with a specific version:
virtualenv --python=/usr/bin/<python3.x> myenv
myenv\Scripts\activate.bat

# [ALTERNATIVE] create conda environment
conda create -n myenv python=<3.x>
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/unimodal](configs/experiment/unimodal)

```bash
python train.py experiment=unimodal/experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```



<div align="center">

# Self-Supervised Multimodal Alignment with Self-Distillation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

We view pairs of multimodal datapoints as augmentations of the same semantic concept and leverage this observation to apply the self-distillation paradigm to the multimodal setting in order to learn a coordinated multimodal representation space. We show that this approach is able to learn a representation space that is more aligned than the one learned by a standard contrastive loss while avoiding the need for negative mining, a cruicial weekness of the contrastive approach.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/marcomoldovan/cross-modal-speech-segment-retrieval
cd cross-modal-speech-segment-retrieval

# install the correct python version
sudo apt-get install python3.10 # Linux, Python 3.7 or higher
brew install python@3.10 #MacOS, Python 3.7 or higher
choco install python --version=3.9 # Windows, Python 3.7-3.9

# create python virtual environment and activate it
python3 -m venv myenv
source myenv/bin/activate

# if you have several version of python you can create a virtual environment with a specific version:
virtualenv --python=/usr/bin/<python3.x> myenv
myenv\Scripts\activate.bat

# [ALTERNATIVE] create conda environment
conda create -n myenv python=<3.x>
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/multimodal](configs/experiment/multimodal)

```bash
python train.py experiment=multimodal/experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```