from custom_yolo.custom_trainer import CustomSegmentationTrainer
from ultralytics.utils.ops import non_max_suppression, scale_coords
from ultralytics.utils.torch_utils import de_parallel
from torch.nn import Sequential, Module
from torch import nn
import torch
from collections import OrderedDict, deque
from typing import List

from copy import deepcopy

from custom_predictor.custom_detection_predictor import CustomSegmentationPredictor

from ultralytics.nn.modules import (
    ## For the bottleneck
    BottleneckCSP, 

    ## For the decoder reconstruct
    Conv, C3k2, A2C2f, 
    
    ## To upsample in decoder
    ConvTranspose)

MODEL_DIR = "train_yolo12n-seg_2025_08_30_00_41_25/yolo12n-seg_data/weights/best.pt"

class YOLOU(Module): 
    def __init__(self, 
                 trainer: CustomSegmentationTrainer, 
                 predictor: CustomSegmentationPredictor,
                 num_classes: int = 1, 
                 image_size: int = 160, 
                 target_modules_indices: List[int] = [0, 1, 4, 6, 8], 
                 yolo_concat_modules_indices: List[int] = [4, 6, 8]): 
        """
        ### WORK IN PROGRESS ### 
        Creates a YOLOU Network with a Pretrain YOLO model
        
        Args: 
            pretrained_yolo (): 
            target_modules_indices (list [int]): 
        
        TODO: Might reduce target module indices for efficiency
        """

        super().__init__()
        ### YOLOv12-Seg Sections
        self.yolo_trainer = trainer
        self.yolo_predictor = predictor
        # self.yolo = self.yolo_trainer.model
        self.yolo = self.yolo_predictor.model.model.model

        print()
        print(type(self.yolo))

        # self.yolo_backbone = pretrained_yolo.model[0:9] 
        # self.yolo_neck1 = pretrained_yolo.model[9:15]
        # self.yolo_neck2 = pretrained_yolo.model[15:21]
        # self.yolo_head = pretrained_yolo.model[21:]

        ### YOLOU Sections
        self.encoder = self.yolo[0:9]
        # self.encoder = self.yolo[0:9]
        self.decoder = self.reconstruct_decoder_from_encoder()

        self.bottleneck = Sequential(
            BottleneckCSP(c1=256, 
                          c2=256, 
                          n=2, 
                          shortcut=True, 
	                          g=1,
                          e=0.5))
        
        self.skip_encoder_indices = set(target_modules_indices)
        # self.skip_decoder_indices = set([abs(target_modules_indices[-1] - item) # ---> Reverses and normalizes the indices
        #                                      for item in target_modules_indices[::-1]])
        self.skip_decoder_indices = set([1, 3, 5, 7, 8])


        ### NOT NEEDED FOR NOW
        # self.YOLO_in_concat_indices1 = set([1, 4])
        # self.YOLO_in_concat_indices2 = set([4]) 
        # self.YOLO_in_concat_indices3 = set([0, 1]) 

        # self.YOLO_out_concat_indices1 = set([4, 6, 8])
        # self.YOLO_out_concat_indices2 = set([2]) 
        # self.YOLO_out_concat_indices3 = set([0, 3]) 
        # self.YOLO_concat = []

        self.skip_connections = []
        self.activation_cache = []

    def hook_fn(self, module, input, output):
        self.activation_cache.append(output.clone().requires_grad_(True))
        print(f"\nSuccessfully cached the output {module}\n")

    def _assign_hooks(self, modules=["0", 
                                     "1",
                                     "3", 
                                     "5", 
                                     "7",
                                     "8"]):
        
    # def _assign_hooks(self, modules=["model.0", 
    #                                  "model.1",
    #                                  "model.3", 
    #                                  "model.5", 
    #                                  "model.7",
    #                                  "model.8"]):
        found = []
        for name, module in self.yolo.named_modules():
            if name in modules:
                module.register_forward_hook(self.hook_fn)
                print(f"Hook registered on: {name} -> {module}")
                found.append(name)
        
        if not found:
            raise ValueError(f"Modules not found in YOLO")

        # # Find the exact module by walking over named_modules
        # for name, module in self.yolo.named_modules():
        #     if name in modules:
        #         module.register_forward_hook(self.hook_fn)
        #         print(f"Hook registered on: {name} -> {module}")
        # raise ValueError(f"Modules not found in YOLO")

    def create_concat_block(self, skip, x):
        conv = Conv(
            c1=skip.size(1)+x.size(1), 
            c2=skip.size(1), 
            k=1, 
            s=1).to(x.device)
        return conv(torch.cat([skip, x], dim=1))

    def YOLO_forward(self, x0: torch.Tensor): 
        """
        YOLOv12-Seg Forward()

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): YOLOv12-Seg output tensor
            (torch.Tensor): cached output (defined in _assign_hook)
        """
        # im = self.yolo_predictor.preprocess(x0)
        # self.yolo_trainer.model.to("cuda")
        self._assign_hooks()
        # preds = self.yolo_trainer.model(im)
        # x2 = self.yolo_predictor.inference(im)

        # detection_outputs, mask_coefs, proto = preds

        # print(detection_outputs[0].size())
        # print(detection_outputs[1].size())
        # print(detection_outputs[2].size())

        # print(mask_coefs.size())
        # print(proto.size())

        # print()
        # print(x2[0].size())
        # print(len(x2[1][0]))
        # print(x2[1][1].size())
        # print(x2[1][2].size())


        # # Handle proto: take last if list
        # if isinstance(proto, (list, tuple)):
        #     proto = proto[-1]

        # import numpy as np

        # # orig_imgs = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(x0.shape[0])]

        # # Postprocess (no_grad!)
        # with torch.no_grad():
        #     results = self.yolo_predictor.postprocess(
        #         preds=x2,    # ← single tensor
        #         img=im,
        #         orig_imgs=x0,
        #         # protos=proto        # ← [B, 32, H, W]
        #     )


        # x = self.yolo_predictor.postprocess(x[0], im, orig_imgs)
        # print(x)
        

        # return x, #self.activation_cache[-1]
        # return self.yolo_trainer.model.predict(x0), self.activation_cache[-1]
        return self.yolo_predictor(x0), self.activation_cache[-1]
        ### TODO: COULD IMPROVE BY POPPING HERE

    def forward(self, x): 
        ### YOLO Forward Pass, x is final YOLOv12-Seg predictions
        yolo_x, backbone_x = self.YOLO_forward(x)
        
        self.skip_connections = self.activation_cache[:-1]

        print(len(yolo_x))

        # # # ### TODO: Implement mask concatenation here, alongside the gating mechanism
        # # # ### (e.g., CBAM or SE blocks)
        self.bottleneck.to("cuda")
        x = self.bottleneck(backbone_x.clone())

        # ### YOLO Mask Concatenation
        # det_outputs = yolo_x[0] # list of 3 tensors (at different scales)
        # mask_coeffs = yolo_x[1] # [1, 32, 525] ← flattened, concatenated coefficients
        # proto = yolo_x[2]       # [1, 32, 40, 40] ← prototype masks

        


        # # print(yolo_x[0][0].size())
        # # print(yolo_x[0][1].size())
        # # print(yolo_x[0][2].size())

        # # print(len(yolo_x[0]))
        # print(yolo_x[1].size())
        # print(yolo_x[2].size())


        # ### Forward through decoder while concatenating skip outputs
        # for i, layer in enumerate(self.decoder.children()):
        #     if i in self.skip_decoder_indices: 
        #         x = self.create_concat_block(self.skip_connections.pop(), x)

        #     x = layer(x)


        return x

    def reverse_module_channels(self, module: nn.Module) -> nn.Module:
        """Reverses the input and output channels of a Conv2d module."""
        if isinstance(module, Conv):
            return ConvTranspose(
                c1=module.conv.out_channels, 
                c2=module.conv.in_channels,
                s=module.conv.stride,
                p=module.conv.padding, 
                bn=True,

                ## To reconstruct the image size of the respective Conv layer
                k=4)
            
        elif isinstance(module, A2C2f):
            return A2C2f(
                c1=module.cv2.conv.out_channels,
                c2=module.cv1.conv.in_channels,
                 
                ## Referencing YOLO12-seg model summary at training start
                a2=True)

        elif isinstance(module, C3k2): 
            return C3k2(
                c1=module.cv2.conv.out_channels, 
                c2=module.cv1.conv.in_channels,
                
                ## Referencing YOLO12-seg model summary at training start
                n=2, 
                c3k=False, 
                e=0.25)
    
    def reconstruct_decoder_from_encoder(self): 
        """
        WORK IN PROGRESS

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        decoder_modules = OrderedDict()        
        
        # Iterate through the encoder layers in reverse order
        for name, module in reversed(list(self.encoder.named_children())):
            reversed_module = self.reverse_module_channels(module)
            decoder_modules[f'decoder_{name}'] = reversed_module
                
        return nn.Sequential(decoder_modules)
    
    def check_encoder_decoder_symmetry(self, backbone_last_index: int): 
        """
        Prints encoder and decoder to check for symmetry

        Args:
            backbone_last_index (int): last index of the yolo_backbone in YOLOv12

        Returns:
            N/A
        """
        for i in range(backbone_last_index): 
            print(f"\n### Comparison {i}:\n{self.encoder[i]}\n")
            print(f"{self.decoder[backbone_last_index - 1 - i]}\n\n")

    def print_yolo_named_modules(self): 
        """
        Prints YOLOv12-Seg named modules. 
        Used for caching "x" in between modules

        Returns:
            N/A
        """
        for name, module in self.yolo.named_modules(): 
            print(f"\n{name}")

if __name__ == "__main__":
    ## Create trainer and predictor instances
    t_args = dict(model=MODEL_DIR,
                data=f"datasets/data.yaml", 
                imgsz=160)
    
    p_args = deepcopy(t_args)
    p_args["save"] = False

    trainer = CustomSegmentationTrainer(overrides=t_args)
    predictor = CustomSegmentationPredictor(overrides=p_args)

    ## Load the model checkpoint
    trainer.setup_model()["model"]         
    predictor.setup_model(MODEL_DIR)

    ## Create YOLOU instance
    model = YOLOU(trainer=trainer, predictor=predictor)
    # model.print_yolo_named_modules()
    zeros = torch.zeros((1, 4, 160, 160))


    x = model(zeros)

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


