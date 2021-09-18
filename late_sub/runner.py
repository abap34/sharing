from utils import info
from trainer import train
from wave_to_spec import transform
import torch

def main(params):
    if params["preprocess"]["skip"]:
        info("skip preprocess...")
    else:
        transform(params["preprocess"])

    train(params["train"])

if __name__ == '__main__':

    params = {
        "preprocess":{
            "skip":True,
            "out_dir_name":"hoge",
            "filter":False,
            "fs":100,
            "nperseg":80
        },
        "train":{
            "data":"filterd",
            "log":True,
            "transform":False,
            "size":32,
            "model_name": "efficientnet_b0",
            "run-id": "exp23",
            "tags": [],
            "optimizer": lambda p: torch.optim.Adam(p, lr=1e-3),
            "freeze":False
        }
    }

    main(params)