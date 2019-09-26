
This repository contains source code for NAACL 2019 paper "An Embarrassingly Simple Approach for Transfer Learning from
Pretrained Language Models" [(Paper link)](https://www.aclweb.org/anthology/N19-1213)


 

# Introduction

This paper presents a simple transfer learning approach that addresses the problem of catastrophic forgetting.
We pretrain a language model and then transfer it to a new model, to which we add a recurrent layer and an attention mechanism. Based on multi-task learning, we use a **weighted sum of losses** (language model loss and classification loss) and fine-tune the pretrained model on our (classification) task.

# Architecture

<p align="center">
<img src="https://user-images.githubusercontent.com/30402550/58558299-19d26680-8229-11e9-893d-99d25c911c7a.png" width="450">
</p>

 
**Step 1**:

- Pretraining of a word-level LSTM-based language model 

**Step 2**: 

- Fine-tuning the language model (LM) on a classification task 

- Use of an auxiliary LM loss

- Employing 2 different optimizers (1 for the pretrained part and 1 for the newly added part)

- Sequentially unfreezing 


#### Reference

```
@inproceedings{chronopoulou-etal-2019-embarrassingly,
    title = "An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models",
    author = "Chronopoulou, Alexandra  and
      Baziotis, Christos  and
      Potamianos, Alexandros",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1213",
    pages = "2089--2095",
}
```

# Prerequisites 

#### Dependencies

- PyTorch version >=0.4.0

- Python version >= 3.6

#### Install Requirements 
**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n siatl python=3
conda activate siatl
```

Install PyTorch ```0.4.0``` with the desired cuda version to use the GPU:

``` conda install pytorch==0.4.0 torchvision -c pytorch```

Then install the rest of the requirements:

```
pip install -r requirements.txt
```

#### Download Data

You can find Sarcasm Corpus V2 [(link)](https://nlds.soe.ucsc.edu/sarcasm2) under ```datasets/```

# Plot visualization

Visdom is used to visualized metrics during training. You should start the server through the command line (using tmux or screen) by typing ```visdom```. Check here for more: https://github.com/facebookresearch/visdom#usage

# Training


In order to train the model, either the LM or the SiATL, you need to run the corresponding python script and pass as an argument a yaml model config. The yaml config specifies all the configuration details of the experiment to be conducted.
To make any changes to a model, change an existing or create a new yaml config file. 

The yaml config files can be found under ```model_configs/``` directory.

#### Use the pretrained Language Model:

``` 
cd checkpoints/
wget https://www.dropbox.com/s/lalizxf3qs4qd3a/lm20m_70K.pt 
```
(Download it and place it in checkpoints/ directory)

#### (Optional) Train a Language Model:

Assuming you have placed the training and validation data under ```datasets/<name_of_your_corpus/train.txt, 
datasets/<name_of_your_corpus/valid.txt``` (check the ```model_configs/lm_20m_word.yaml```'s data section), you can train a LM. 

See for example:

``` python models/sent_lm.py -i lm_20m_word.yaml ```

#### Fine-tune the Language Model on the labeled dataset, using an auxiliary LM loss, 2 optimizers and sequential unfreezing, as described in the paper:

To fine-tune it on the Sarcasm Corpus V2 dataset:

``` python models/run_clf.py -i SCV2_aux_ft_gu.yaml --aux_loss --transfer```

- ``-i``: Configuration yaml file (under ``model_configs/``)
- ``--aux_loss``: You can choose if you want to use an **auxiliary LM** loss 
- ``--transfer``: You can choose if you want to use a **pretrained LM** to initalize
the embedding and hidden layer of your model. If not, they will be randomly initialized
