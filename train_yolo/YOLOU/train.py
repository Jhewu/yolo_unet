
from torch.nn import Sequential, Module
from torch import nn
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torch.amp import GradScaler

import os
from collections import OrderedDict
from typing import List, Tuple
from copy import deepcopy
import time

from custom_predictor.custom_detection_predictor import CustomSegmentationPredictor
from custom_yolo.custom_trainer import CustomSegmentationTrainer

from PIL import Image
import numpy as np

from ultralytics.nn.modules import (
    ## For the bottleneck
    BottleneckCSP, 

    ## For the decoder reconstruct
    Conv, C3k2, A2C2f, 
    
    ## To upsample in decoder
    ConvTranspose)



    






if __name__ == "__main__":
    MODEL_DIR = "train_yolo12n-seg_2025_08_30_00_41_25/yolo12n-seg_data/weights/best.pt"

    ## Create trainer and predictor instances
    t_args = dict(model=MODEL_DIR,
                data=f"datasets/data.yaml", 
                imgsz=160)
    
    p_args = deepcopy(t_args)
    p_args["save"] = False      # --> Needs to be False for predictor
                                #     Otherwise error

    trainer = CustomSegmentationTrainer(overrides=t_args)
    predictor = CustomSegmentationPredictor(overrides=p_args)

    ## Load the model checkpoint
    trainer.setup_model()["model"]         
    predictor.setup_model(MODEL_DIR)

    ## Create YOLOU instance
    model = YOLOU(trainer=trainer, predictor=predictor)
    zeros = torch.zeros((1, 4, 160, 160))

    img = Image.open("BraTS-SSA-00041-00036-t1c.png")
    
    x = model(img)

    # model.check_encoder_decoder_symmetry(9)





    # print(x)
    
    

    

    # x = predictor("BraTS-SSA-00041-00036-t1c.png")
    # # x = predictor(zeros)
    # import numpy as np
    # print(np.unique(x[0].masks.data.cpu()))
    # print(x[0].masks.data.cpu())

    # x = pretrained_yolo(zeros)


    # x = model(zeros)

    # print(x)

    # print(model.activation_cache[0])
    # print(x[1])

    # print(type(model.activation_cache[0][0]))
    # print(type(x[0]))

    # print(torch.equal(x[0], model.activation_cache[0][0]))
    # print(model.activation_cache[0] is x)

    ## Obtain the pretrain until last layer
    # no_head = pretrained_yolo.model[:-1]

    # for i, layer in enumerate(pretrained_yolo.model[-1].named_children()): 
    #     print(i)
    #     print(layer)
    #     print()

    # x = pretrained_yolo(zeros)

    # print(x[0][0].size())
    # print(x[1].size())
    # print(x[2].size())

    # # print(len(x[0]))
    # # print(len(x[1]))
    # # print(len(x[2]))

    # print(pretrained_yolo.model[-1])

    # import matplotlib.pyplot as plt
    # plt.imshow(x[0][0].detach().numpy())
    # plt.show()
    
    # outputs = model(zeros)

    # print(x)


