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
from PIL import Image


from ultralytics.nn.modules import (
    ## For the bottleneck
    BottleneckCSP, 

    ## For the decoder reconstruct
    Conv, C3k2, A2C2f, 
    
    ## To upsample in decoder
    ConvTranspose)

from typing import Tuple

MODEL_DIR = "train_yolo12n-seg_2025_08_30_00_41_25/yolo12n-seg_data/weights/best.pt"

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root_path: str, 
                 image_path: str, mask_path: str, 
                 image_size: int, subsample: float = 1.0):
        """
        Create Local Dataset for Image Segmentation

        Args:
            root_path   (str): dataset root path where images and masks directories are present
            image_path  (str): images path (relative to root_path)
            mask_path   (str): masks path (relative to root_path)
            image_size  (int): img_size x img_size to load images
            subsample (float): loads only a subset of images
        """
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i 
                              for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i 
                             for i in os.listdir(root_path+f"/{mask_path}/")])

        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        # Subsample if implemented
        self.images = self.images[:int(len(self.images)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[index]).convert("RGBA")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

class ECA(nn.Module):
    def __init__(self, k_size: int = 3):
        super(ECA, self).__init__()
        """
        Constructs a ECA module. Efficient Channel Attention for Conv

        Args: 
            k_size (int): kernel size for Conv1d
        """
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x (torch.tensor): input tensor (after mask + x concat)
        Returns:
            (torch.tensor)  : output tensor (same dimensions as input tensor but with attention applied)
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(y)

        return x * y.expand_as(x)

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, mode: str = 'bilinear'):
        """
        Spatial Transformer Module that predicts affine transformation 
        from backbone features ONLY (no mask concatenation).
        Can be applied to features or input images.

        Args: 
            in_channels (int): input channels
            mode        (str): interpolation method
        """
        super().__init__()
        self.mode = mode

        # Localization network: predicts 6 parameters for 2x3 affine matrix
        self.loc_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, kernel_size=1)  # Output: [B, 6, 1, 1]
        )

        # Initialize to identity transform
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x (torch.tensor): input tensor [B, C, H, W] — could be image or feature map
        Returns:
            transformed (torch.tensor): warped version of x
            theta       (torch.tensor): affine matrix [B, 2, 3]
        """
        # Predict affine parameters from global pooling
        theta_flat = self.loc_net(x).squeeze(-1).squeeze(-1)  # [B, 6]
        theta = theta_flat.view(-1, 2, 3)  # [B, 2, 3]

        # Create sampling grid
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)

        # Sample input
        transformed = torch.nn.functional.grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode='border',
            align_corners=False
        )

        return transformed, theta

class YOLOU(Module): 
    def __init__(self, 
                 trainer: CustomSegmentationTrainer, 
                 predictor: CustomSegmentationPredictor,
                 num_classes: int = 1, 
                 image_size: int = 160, 
                 target_modules_indices: List[int] = [0, 1, 3, 5, 7]): 
        """
        ### WORK IN PROGRESS ### 
        Creates a YOLOU Network with a Pretrain YOLOv12-Seg model
        Idea: rough masks (YOLOv12-Seg) --> fine masks (YOLOU)
        
        Args: 
            trainer   (CustomSegmentationTrainer): Custom YOLO segmentation trainer allowing 4-channels
            predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor allowing 4-channels
            target_modules_indices   (list [int]): list of indices to add skip connections (in YOLOv12-Seg every downsample)
        
        TODO: Might reduce target module indices for efficiency
        """

        super().__init__()
        ### YOLOv12-Seg Sections
        self.yolo_trainer = trainer
        self.yolo_predictor = predictor
        self._yolo_seq = self.yolo_predictor.model.model.model

        ### YOLOU Sections
        self.encoder = self._yolo_seq[0:9]
        self.decoder = self._construct_decoder_from_encoder()
        self.bottleneck1 = Sequential(
            BottleneckCSP(c1=256, 
                          c2=256, 
                          n=1, 
                          shortcut=True, 
	                          g=1,
                          e=0.5)).to("cuda")
        self.bottleneck2 = Sequential(
            BottleneckCSP(c1=512, 
                          c2=256, 
                          n=1, 
                          shortcut=True, 
	                          g=1,
                          e=0.5)).to("cuda")
        self.stn = None
        self.eca = None
        
        ### YOLOU Helper
        self.skip_encoder_indices = set(target_modules_indices)
        self.skip_decoder_indices = set([abs(target_modules_indices[-1] - item + 1) # ---> Reverses and normalizes the indices
                                             for item in target_modules_indices[::-1]])

        self.skip_connections = []
        self.activation_cache = []

    def _hook_fn(self, module, input, output):
        """
        Forward hook, once activate appends the output to
        self.activation_cache
        """
        self.activation_cache.append(output)
        print(f"\nSuccessfully cached the output {module}\n")

    def _assign_hooks(self, modules: list[str] = ["0", 
                                                "1",
                                                "3", 
                                                "5", 
                                                "7",
                                                "8"]):
        """
        Assigns forward hooks for YOLOv12-Seg forward
        Depends on self._hook_fn()

        Args:
            modules (list[str]): List containing the names of the modules
        """        
        found = []
        for name, module in self._yolo_seq.named_modules():
            if name in modules:
                module.register_forward_hook(self._hook_fn)
                print(f"Hook registered on: {name} -> {module}")
                found.append(name)
        
        if not found:
            raise ValueError(f"Modules not found in YOLO")

    def _create_concat_block(self, skip: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Creates encoder-to-decoder concat blocks during the decoder step in forward()

        Args:
            skip (torch.tensor): Encoder skip tensor
            x    (torch.tensor): Decoder x tensor from bottleneck

        Returns:
            (torch.tensor): concated tensor of size [B, skip_C + x_C, H, W]
        """
        conv = Conv(
            c1=skip.size(1)+x.size(1), 
            c2=skip.size(1), 
            k=1, 
            s=1).to(x.device)
        return conv(torch.cat([skip, x], dim=1))

    def YOLO_forward(self, x: torch.tensor) -> torch.tensor: 
        """
        YOLOv12-Seg Forward() used at first step of YOLOU

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            (torch.tensor): YOLOv12-Seg masks in batches
            (torch.tensor): cached backbone output (defined in _assign_hook)
        """
        self._assign_hooks()
        results = self.yolo_predictor(x)

        # Sums the masks and stack it
        mask_batch = []
        for result in results: 
            if result.masks is not None: 
                mask_sum = torch.sum(result.masks.data, dim = 0)
            else: 
                mask_sum = torch.zeros(result.orig_shape[0], result.orig_shape[1])
            mask_batch.append(mask_sum.unsqueeze(0))

        return torch.stack(mask_batch), self.activation_cache.pop()
    
    def _STN_forward(self, x: torch.tensor) -> torch.tensor: 
        """
        Adaptive STN Forward()
        STN Module Learns Affine transform parameters
        Automatically Set Channel

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            (torch.tensor): Spatially transformed tensor
        """
        self.stn = SpatialTransformer(in_channels=x.size()[1]).to("cuda")
        return self.stn(x)
    
    def _concat_masks_forward(self, masks: torch.tensor, x: torch.tensor) -> torch.tensor: 
        """
        Adapt YOLO predicted, and concantenate
        with CSPNet Bottleneck features while 
        applying ECA-Net attention

        Args:
            masks (torch.tensor): Masks tensor (B,1,160,160)
            x     (torch.tensor): Input tensor

        Returns:
            (torch.tensor): 
        """
        mask = torch.nn.functional.interpolate(masks, size=(5, 5), mode='bilinear').to("cuda")  # -> [B, 1, 5, 5]
        proj_conv = nn.Conv2d(1, 256, kernel_size=1).to("cuda")                                 
        masks = proj_conv(mask)                                                                 # -> [B, 256, 5, 5]
        concat = torch.cat([masks, x], dim=1).to("cuda")                                        # -> [B, 512, 5, 5]
        self.eca = ECA().to("cuda")                                                             # Applying masks attention
        return self.eca(concat)
    
    # def train(self): 
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # train_dataset = SegmentationDataset(DATA_PATH, "images/train", "labels/train", IMAGE_SIZE)
    # val_dataset = SegmentationDataset(DATA_PATH, "images/test", "labels/test", IMAGE_SIZE)

    # train_dataloader = DataLoader(dataset=train_dataset,
    #                             batch_size=BATCH_SIZE,
    #                             shuffle=True)
    # val_dataloader = DataLoader(dataset=val_dataset,
    #                             batch_size=BATCH_SIZE,
    #                             shuffle=True)

    # model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # # Grab the original conv1
    # orig_conv1 = model.backbone.conv1

    # # Build a new conv that takes 4 channels
    # new_conv1 = torch.nn.Conv2d(
    #     in_channels=4,                    # <-- changed
    #     out_channels=orig_conv1.out_channels,
    #     kernel_size=orig_conv1.kernel_size,
    #     stride=orig_conv1.stride,
    #     padding=orig_conv1.padding,
    #     bias=orig_conv1.bias is not None   # preserve bias flag
    # )

    # # --- initialize the new conv ---------------------------------
    # with torch.no_grad():
    #     # Copy the pre‑trained weights for the first 3 channels
    #     new_conv1.weight[:, :3] = orig_conv1.weight

    #     # Decide how to initialise the 4th channel
    #     # 1) Zero init (most common)
    #     new_conv1.weight[:, 3:4] = torch.zeros_like(orig_conv1.weight[:, :1])

    # # Swap the conv in the model
    # model.backbone.conv1 = new_conv1

    # # Replace the final 1x1 conv to output 1 channel
    # new_classifier = torch.nn.Sequential(
    #     # keep everything before the final conv
    #     *list(model.classifier.children())[:-1],          # all layers except the last conv
    #     torch.nn.Conv2d(
    #         in_channels=256,
    #         out_channels=1,        # <‑‑ change here
    #         kernel_size=1,
    #         stride=1,
    #         padding=0
    #     )
    # )

    # # Assign the new head back to the model
    # model.classifier = new_classifier

    # if LOAD_AND_TRAIN: 
    #     model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    # summary(model, input_size=(BATCH_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE))

    # # Initialize the optimizer, to adjust the parameters of a model and minimize the loss function
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # if LOAD_AND_TRAIN:
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.3, patience=int(PATIENCE * 0.5), verbose=True)
    # else: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # # For Mixed-Precision Training
    # scaler = GradScaler("cuda")

    # # Initialize variables for callbacks
    # history = dict(train_loss=[], val_loss=[], train_dice_metric=[], val_dice_metric=[])
    # best_val_loss = float("inf")

    # dest_dir = f"runs/unet_{GetCurrentTime()}" 
    # model_dir = os.path.join(dest_dir, "weights")
    # CreateDir(model_dir)

    # # Initialize local patience variable for early stopping
    # patience = 0

    # combined_loss = DiceFocalTverskyLoss()

    # for epoch in tqdm(range(EPOCHS)):
    #     model.train()

    #     start_time = time.time()
    #     train_running_loss = 0
    #     train_running_dice_metric = 0

    #     if MIX_PRECISION:
    #         for idx, img_mask in enumerate(tqdm(train_dataloader)):
    #             with torch.amp.autocast(device_type="cuda"): 
    #                 img = img_mask[0].float().to(device)
    #                 mask = img_mask[1].float().to(device)

    #                 y_pred = model(img)["out"]
    #                 loss = combined_loss(y_pred, mask)
    #                 metric = dice_metric(y_pred, mask)

    #             optimizer.zero_grad()
    #             scaler.scale(loss).backward()

    #             # Unscales the gradients of optimizer's assigned params in-place
    #             scaler.unscale_(optimizer)

    #             # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #             # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    #             # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    #             scaler.step(optimizer)

    #             # Updates the scale for next iteration.
    #             scaler.update()

    #             train_running_loss += loss.item()
    #             train_running_dice_metric += metric.item()

    #     else:
    #         for idx, img_mask in enumerate(tqdm(train_dataloader)):
    #             img = img_mask[0].float().to(device)
    #             mask = img_mask[1].float().to(device)

    #             y_pred = model(img)["out"]
    #             optimizer.zero_grad()

    #             loss = dice_loss(y_pred, mask)
    #             metric = dice_metric(y_pred, mask)

    #             train_running_loss += loss.item()
    #             train_running_dice_metric += metric.item()
                
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()

    #     end_time = time.time()
    #     train_loss = train_running_loss / (idx + 1)
    #     train_dice_metric = train_running_dice_metric / (idx + 1)

    #     model.eval()
    #     val_running_loss = 0
    #     val_running_dice_metric = 0

    #     with torch.no_grad():
    #         for idx, img_mask in enumerate(tqdm(val_dataloader)):
    #             img = img_mask[0].float().to(device)
    #             mask = img_mask[1].float().to(device)
                
    #             y_pred = model(img)["out"]
    #             loss = dice_loss(y_pred, mask)
    #             val_metric = dice_metric(y_pred, mask)

    #             val_running_loss += loss.item()
    #             val_running_dice_metric += val_metric.item()

    #         val_loss = val_running_loss / (idx + 1)
    #         val_dice_metric = val_running_dice_metric / (idx + 1)
        
    #     # Update the scheduler
    #     if LOAD_AND_TRAIN:
    #         scheduler.step(val_loss)
    #     else: scheduler.step()

    #     history["train_loss"].append(train_loss)
    #     history["val_loss"].append(val_loss)
    #     history["val_dice_metric"].append(val_dice_metric)
    #     history["train_dice_metric"].append(train_dice_metric)

    #     if val_loss < best_val_loss: 
    #         if (best_val_loss - val_loss) > 1e-3:
    #             print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
    #             best_val_loss = val_loss
    #             torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
    #             patience = 0
    #         else: 
    #             print(f"Validation loss improved slightly from {best_val_loss:.4f} to {val_loss:.4f}, but not significantly enough to save the model.")
    #             if epoch+1 >= EARLY_STOPPING_START: 
    #                 patience+=1
    #     else:
    #         if epoch+1 >= EARLY_STOPPING_START: 
    #             patience+=1
        
    #     history_df = pd.DataFrame(history)
    #     history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

    #     print("-"*30)
    #     print(f"This is Patience {patience}")
    #     print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
    #     print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(device) * 1e-9:.4f}")
    #     print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    #     print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    #     print(f"Train DICE Score EPOCH {epoch+1}: {train_dice_metric:.4f}")
    #     print(f"Valid DICE Score EPOCH {epoch+1}: {val_dice_metric:.4f}")
    #     print("-"*30)

    #     if patience >= PATIENCE: 
    #         print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience}, terminating training...")
    #         break

    # torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "last.pth")))
    # plot_loss_curves(history, dest_dir)

    def forward(self, x: torch.tensor) -> torch.tensor: 
        """
        Main forward step of YOLOU 

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]

        Returns:
            x (torch.tensor): Output tensor [B, 4, H, W]
        """
        # encoder
        yolo_x, backbone_x = self.YOLO_forward(x)
        
        # bottleneck
        backbone_x, theta = self._STN_forward(backbone_x.clone())
        x = self.bottleneck1(backbone_x)
        x = self._concat_masks_forward(yolo_x, x)
        x = self.bottleneck2(x)

        # decoder
        # COMMENT: for some reason self.yolo_predictor(x) triggers forward hooks twice, therefore
        #          take the middle and +1, because we popped at YOLO_forward()
        self.skip_connections = self.activation_cache[(len(self.activation_cache)//2)+1:]

        for i, layer in enumerate(self.decoder.children()):
            layer.to("cuda")
            if i in self.skip_decoder_indices: 
                a = self.skip_connections.pop()
                x = self._create_concat_block(a, x)

            x = layer(x)
        return x

    def _reverse_module_channels(self, module: nn.Module) -> nn.Module:
        """
        In essence, it reverses the input and output of a nn.Module
        present in YOLOv12 backbone. If it's a Conv module then it uses
        a respective ConvTranspose module. It's used to create the
        decoder of YOLOU from the YOLOv12 backbone

        Args:
            module (nn.Module): Module from YOLOv12 Seg backbone

        Returns:
            (nn.Module): Corresponding module to built the decoder
        """

        if isinstance(module, Conv):
            return ConvTranspose(
                c1=module.conv.out_channels, 
                c2=module.conv.in_channels,
                s=module.conv.stride,
                p=module.conv.padding, 
                bn=True,

                # To reconstruct the image size of the respective Conv layer
                k=4)
            
        elif isinstance(module, A2C2f):
            return A2C2f(
                c1=module.cv2.conv.out_channels,
                c2=module.cv1.conv.in_channels,
                 
                # Referencing YOLO12-seg model summary at training start
                a2=True)

        elif isinstance(module, C3k2): 
            return C3k2(
                c1=module.cv2.conv.out_channels, 
                c2=module.cv1.conv.in_channels,
                
                # Referencing YOLO12-seg model summary at training start
                n=2, 
                c3k=False, 
                e=0.25)
    
    def _construct_decoder_from_encoder(self) -> nn.Sequential:
        """
        Construct YOLOU decoder from YOLOv12 Seg backbone. 
        Depends on self._reverse_module_channels()

        Returns:
            (nn.Sequential): Respective decoder sequential list
        """
        decoder_modules = OrderedDict()        
        
        # Iterate through the encoder layers in reverse order
        for name, module in reversed(list(self.encoder.named_children())):
            reversed_module = self._reverse_module_channels(module)
            decoder_modules[f'decoder_{name}'] = reversed_module
                
        return nn.Sequential(decoder_modules)
    
    def check_encoder_decoder_symmetry(self, backbone_last_index: int): 
        """
        Prints encoder and decoder to check for symmetry

        Args:
            backbone_last_index (int): Last index of the YOLO backbone in YOLOv12 Seg
        """
        for i in range(backbone_last_index): 
            print(f"\n### Comparison {i}:\n{self.encoder[i]}\n")
            print(f"{self.decoder[backbone_last_index - 1 - i]}\n\n")

    def print_yolo_named_modules(self): 
        """
        Prints YOLOv12-Seg named modules. 
        Used for caching "x" in between modules
        """
        for name, module in self._yolo_seq.named_modules(): 
            print(f"\n{name}")

if __name__ == "__main__":
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


