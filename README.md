
This repository contains source code for NAACL 2019 paper "An Embarrassingly Simple Approach for Transfer Learning from
Pretrained Language Models" [(Paper link)](https://arxiv.org/abs/1902.10547)

 
The modules used to pretrain the LM and fine-tune it to 5 downstream classification tasks will be added shortly.

# Introduction

This paper presents present a simple transfer learning approach that addresses the problem of catastrophic forgetting.
We pretrain a language model and then transfer it to a new model, to which we add a recurrent layer and an attention mechanism. Based on multi-task learning, we use a weighted sum of losses and fine-tune the pretrained model on our (classification) task.

### Architecture

<img src="https://github.com/alexandra-chron/siatl/images/siatl.png" width="380">


**Step 1**: Pretraining of a word-level LSTM-based language model 

**Step 2**: Fine-tuning the language model (LM) on a classification task. Use of an auxiliary LM loss, 2 different optimizers (1 for the pretrained part and 1 for the newly added part), sequentially unfreezing method.


#### Reference

```
@article{DBLP:journals/corr/abs-1902-10547,
  author    = {Alexandra Chronopoulou and
               Christos Baziotis and
               Alexandros Potamianos},
  title     = {An Embarrassingly Simple Approach for Transfer Learning from Pretrained
               Language Models},
  booktitle = {Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL:HLT)},
  address   = {Minneapolis, USA},
  month     = {June},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.10547}
}
```

# Prerequisites 

#### Dependencies

- PyTorch version >=1.0.0

- Python version >= 3.6

#### Install Requirements 
** Create Enviornment (Optional): **  Ideally, you should create a conda environment for the project.

```
conda create -n siatl python=3
conda activate siatl
```

Install PyTorch ```1.1.0``` with the desired cuda version to use the GPU:

``` conda install pytorch torchvision -c pytorch```

Then install the rest of the requirements:

```
pip install -r requirements.txt
```

#### Download Data

You can find Sarcasm Corpus V2 [(link)](https://nlds.soe.ucsc.edu/sarcasm2) under ```datasets/```


# Training


In order to train the model, either the LM or the SiATL, you need to run the corresponding python script and pass as an argument a yaml model config. The yaml config specifies all the configuration details of the experiment to be conducted.
To make any changes to a model, change an existing or create a new yaml config file. 

The yaml config files can be found under ```model_configs/``` directory.

### Train a Language Model:

Assuming you have placed the training and validation data under ```datasets/<name_of_your_corpus/train.txt``` and 
```datasets/<name_of_your_corpus/valid.txt``` (check the ```model_configs/lm_20m_word.yaml```'s data section), you can train a LM. See for example:

``` python models/sent_lm.py -i lm_20m_word.yaml ```

### Fine-tune the Language Model on the labeled dataset, using an auxiliary LM loss, 2 optimizers and sequential unfreezing, as described in the paper:

``` python models/run_clf.py -i SCV2_aux_ft_gu.yaml ```

