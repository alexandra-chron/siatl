import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "model_configs")

TRAINED_PATH = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = os.path.join(BASE_DIR, 'datasets')

EXP_DIR = os.path.join(BASE_DIR, 'experiments')

MODEL_DIRS = ["models", "modules", "utils"]

EXP = {

}

VIS = {
    "enabled": True,
    "server": "http://localhost",
    "port": 8097,
    "base_url": "/",
    "http_proxy_host": None,
    "http_proxy_port": None
}
