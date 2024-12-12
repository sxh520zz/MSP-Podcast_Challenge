import os
import torch
import time
import numpy as np
import json
def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_ssl_type(ssl_type):
    ssl_book={
        "wavlm-large": "microsoft/wavlm-large"
    }
    return ssl_book.get(ssl_type, None)