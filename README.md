# Cross-Modal Speech Segment Retrieval

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

We tackle the problem of learning a multimodal representation space for language in the form of text as well as speech. We contrastively align semantically similar text and speech segments in the representation space in order to enable cross-modal retrieval of speech segments given a text query and vice versa.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/marcomoldovan/cross-modal-speech-segment-retrieval
cd cross-modal-speech-segment-retrieval

# [OPTIONAL] create python virtual environment
# Requires Python 3.7-3.9 on Windows or Python 3.7 or higher on Linux and MacOS
python3 -m venv myenv # uses default python version
virtualenv --python=/usr/bin/<python3.x> myenv # to specify python version
myenv\Scripts\activate.bat # for Windows
source myenv/bin/activate # for Linux or MacOS

# [ALTERNATIVE] create conda environment
conda create -n myenv python=3.8
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

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
