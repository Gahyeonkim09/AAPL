import os.path as osp
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (MetricMeter, AverageMeter, load_checkpoint, load_pretrained_weights)
from dassl.data.transforms import build_transform
import datasets

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from matplotlib import pyplot as plt
import numpy as np
import pdb
import time
import datetime
from scipy.stats import truncnorm
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
import PIL
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms as transforms
import random
import json

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from itertools import combinations

import cv2

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class ClipTextEncoder(nn.Module):
    """"""
    def __init__(self, clip: nn.Module) -> None:
        super().__init__()
        # self.dtype = clip.dtype
        self.encoder = clip.transformer
        self.layer_norm = clip.ln_final
        self.projection = clip.text_projection

        self.positional_embeds = (
            clip.positional_embedding.type(clip.dtype).cuda()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        outputs = inputs + self.positional_embeds # .type(self.dtype)
        outputs = torch.permute(outputs, (1, 0, 2))
        outputs = self.encoder(outputs)
        outputs = torch.permute(outputs, (1, 0, 2))
        outputs = self.layer_norm(outputs)# .type(self.dtype)

        # Positions of '[EOS]' tokens.
        cols = torch.argmax(tokens, dim=-1)
        rows = torch.arange(outputs.size(0))
        outputs = outputs[rows, cols]
        outputs = outputs @ self.projection

        return outputs

# get the random image pair except for the original image's label.    
class GetImagePair(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        train_dataset_name = cfg.DATASET.NAME                   
        dataset = eval('datasets.'+train_dataset_name)(cfg)     
        self.tracker = dataset.tracker                      
        self.transform = build_transform(cfg)       
    
    # choose the random label(except for the original label)
    def read_image(self, label):
        labels = [j for j in range(0,int(len(self.tracker)))]
        labels.remove(label.item())
        label_2 = random.choice(labels)
        img2_path = random.choice(self.tracker[label_2])                    
        img2 = Image.open(img2_path).convert("RGB")     # 이미지 읽기 
        return img2, label_2, img2_path

    def forward(self, label):
        img2, label_2, img2_path = self.read_image(label)         
        trans_img2 = self.transform(img2)
        label_2 = torch.tensor([label_2]).cuda()
        return trans_img2, label_2, img2_path
    
class ClipFeatureExtractor(nn.Module):
    def __init__(self, cfg, model: str, device: torch.device) -> None:
        super().__init__()
        model, _ = clip.load(model, device=device)
        # Cache data type.
        self.dtype = model.dtype
        # Cache device.
        self.device = device
        # Cache logit scaler.
        self.scaler = (
            torch.exp(model.logit_scale)
        )

        # Vision Transformer model.
        self.image_encoder = model.visual

        # Transformer decoder model.
        self.text_encoder = model.transformer
        self.text_project = model.text_projection 
        self.layer_norm = model.ln_final

        # Pre-computed position embeddings.
        self.position_embeds = (
            model.positional_embedding.type(model.dtype)
        )
        
    # @torch.no_grad()
    def normalize(self, embeds: torch.Tensor) -> torch.Tensor:
        return embeds / torch.norm(embeds, dim=1, keepdim=True)

    # @torch.no_grad()
    def encode_texts(self, embeds, tokens):
        # Manually encoder position information.
        outputs = embeds + self.position_embeds

        # Pass through Transformer encoder model.
        outputs = torch.permute(outputs, (1, 0, 2))
        outputs = self.text_encoder(outputs)
        outputs = torch.permute(outputs, (1, 0, 2))
        outputs = self.layer_norm(outputs)

        # Positions of '[EOS]' tokens.
        cols = torch.argmax(tokens, dim=-1)
        rows = torch.arange(outputs.size(0))
        outputs = outputs[rows, cols]
        outputs = outputs @ self.text_project
        outputs = self.normalize(embeds=outputs)

        return outputs

    # @torch.no_grad()
    def encode_images(self, images):
        images = images.type(self.dtype)
        outputs = self.image_encoder(images)
        outputs = self.normalize(embeds=outputs)
        return outputs

# Augment the images(image1, image2)    
class DistortImage(nn.Module):
    def __init__(self):
        super().__init__()
    # It is important that the p=1.0 for the Random augmentation functions.    
    # Color Jittering Augmentation(R,G,B)
    def color_jitter_red(self, image, brightness=1.5, contrast=1.0, saturation=1.0, hue=0.2):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(image)
    def color_jitter_green(self, image, brightness=1.0, contrast=1.0, saturation=1.5, hue=0):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(image)
    def color_jitter_blue(self, image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(image)

    # Grayscale Augmentation
    def grayscale(self, image):
        transform = transforms.Grayscale(num_output_channels=3)
        return transform(image)

    # Gaussian Blur Augmentation
    def gaussian_blur(self, image, kernel_size=3):
        transform = transforms.GaussianBlur(kernel_size=kernel_size)
        return transform(image)

    # Gaussian Noise Augmentation
    def gaussian_noise(self, image, mean=0, std=1):
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        return noisy_image

    # Flip Augmentation (Horizontal, Vertical)
    def horizontal_flip(self, image):
        transform = transforms.RandomHorizontalFlip(p=1.0)
        return transform(image)
    def vertical_flip(self, image):
        transform = transforms.RandomVerticalFlip(p=1.0)
        return transform(image)

    # Rotation(90, 180, 270) Augmentation
    def rotation_90(self, image, degrees = (90,90)):
        transform = transforms.RandomRotation(degrees)
        return transform(image)
    def rotation_180(self, image, degrees = (180,180)):
        transform = transforms.RandomRotation(degrees)
        return transform(image)
    def rotation_270(self, image, degrees = (270, 270)):
        transform = transforms.RandomRotation(degrees)
        return transform(image)
     
    # sobel filtering
    def sobel_filter(self, image):
        image = np.array(image.cpu())
        if image.ndim == 3:
            output_np = np.zeros_like(image)
            num_images = 1
        elif image.ndim == 4:  # 이미지 배치인 경우
            output_np = np.zeros_like(image)
            num_images = image.shape[0]
        for i in range(num_images):
            for c in range(3):  # RGB 채널
                if num_images > 1:  # 이미지 배치인 경우
                    img = image[i,c,:,:]
                else:  # 단일 이미지인 경우
                    img = image[c,:,:]
                sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                sobel_filtered = np.sqrt(sobel_x**2 + sobel_y**2)

                if num_images > 1:
                    output_np[i,c,:,:] = sobel_filtered
                else:
                    output_np[c,:,:] = sobel_filtered      

        sobel_combined = torch.from_numpy(output_np)
        return sobel_combined.cuda()
    
    # crop and resize
    def crop(self, image, top=30, left=30, height=224, width=224):
        transform = transforms.functional.crop(image, top, left, height, width)
        return transform
    def cutout(self, image):
        transform = transforms.RandomErasing(p=1.0, scale=(0.5, 0.5), ratio=(0.3, 3.3), value=0)
        return transform(image)
            
    def distort_1(self, image, augment_type_1):
        if hasattr(self, augment_type_1) and callable(getattr(self, augment_type_1)):
            augmentation_func_1 = getattr(self, augment_type_1)
            distorted_img_A = augmentation_func_1(image)
        return distorted_img_A

    def forward(self, image , augment_type_1, augment_type_2): 
        # get the random augmnetation type in 'str', match it to the actual function and apply it.
        if hasattr(self, augment_type_1) and callable(getattr(self, augment_type_1)):
            augmentation_func_1 = getattr(self, augment_type_1)
            distorted_img_A = augmentation_func_1(image)

        if hasattr(self, augment_type_2) and callable(getattr(self, augment_type_2)):
            augmentation_func_2 = getattr(self, augment_type_2)
            distorted_img_B = augmentation_func_2(image)           
            
        distorted_imgs = torch.stack([distorted_img_A, distorted_img_B])     # return the distorted images.
        return distorted_imgs
   
# Difference Vector Extractor
class DiVE(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.get_image_pair = GetImagePair(cfg)      # get a random pair image
        self.clip_model = clip_model
        vis_dim = clip_model.visual.output_dim
        self.cfg = cfg
        self.classnames = classnames  
        self.n_cls = len(classnames)
        n_attr = cfg.TRAINER.COCOOP.N_ATTR
        dtype = clip_model.dtype
        self.augment_types = ['color_jitter_red', 'color_jitter_green', 'color_jitter_blue', 'grayscale', 'gaussian_noise', 'gaussian_blur','horizontal_flip', 
                    'vertical_flip', 'rotation_90', 'rotation_180', 'rotation_270', 'crop', 'cutout', 'sobel_filter']
        
        self.image_distortion = DistortImage()
        self.augtype_meta_tokens = []                       # for the augmetation analysis

        learnable_attribute = torch.empty(n_attr, vis_dim, dtype=dtype)
        nn.init.normal_(learnable_attribute, std=0.02)
        learnable_attribute = " ".join(["X"] * n_attr)        
        
        self.learnable_attribute = learnable_attribute
        classnames = [name.replace("_", " ") for name in classnames]
        prompt = [learnable_attribute + " " + name + "." for name in classnames] 
        tokenized_attr = torch.cat([clip.tokenize(p) for p in prompt])
        tokenized_cls = torch.cat([clip.tokenize(c) for c in classnames]) 

        self.tokenized_cls = tokenized_cls
        self.tokenized_attr = tokenized_attr 
        
        with torch.no_grad():
            attr_embedding= clip_model.token_embedding(tokenized_attr).type(dtype)     # [n_cls, max_token_len, dim]
            cls_embedding = clip_model.token_embedding(tokenized_cls).type(dtype)      # [n_cls, max_token_len, dim]

        self.attr_embedding = attr_embedding
        self.cls_embedding = cls_embedding
        self.token_prefix = attr_embedding[:, :1, :]                # SOS
        self.token_suffix = attr_embedding[:, 1 + n_attr :, :]      # EOS        
    
        attr_vectors = attr_embedding[0, 1 : 1 + n_attr, :]         # make the attribute vectors learnable
        self.learnable_attr = nn.Parameter(attr_vectors)           
        self.logit_scale = clip_model.logit_scale
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim//16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim//16, vis_dim)),
        ]))              
        
        for param in self.meta_net.parameters():
            param.data = param.type(dtype)

        model = cfg.MODEL.BACKBONE.NAME
        device = torch.device("cuda")
        self.extractor = ClipFeatureExtractor(cfg, model=model, device=device)
         
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

    def triplet_loss(self, anchor_embed, positive_embed, negative_embed, margin=0.2):
        pos_distance = torch.sum((anchor_embed - positive_embed).pow(2), dim=-1)
        neg_distance = torch.sum((anchor_embed - negative_embed).pow(2), dim=-1)
        loss = torch.max(pos_distance - neg_distance + margin, torch.tensor(0.0).to(anchor_embed.device))
        loss = loss.mean()
        pos_distance = pos_distance.mean()
        neg_distance = neg_distance.mean()
        return loss, pos_distance, neg_distance      # Return the mean loss for the batch
    
    # make a prompt with meta-tokens and the learnable vectors.
    def prompt_maker(self, meta_tokens, image_features):
        logit_scale = self.logit_scale.exp()  
        learnable_attr = self.learnable_attr.cuda()
        prefix = self.token_prefix.cuda()
        suffix = self.token_suffix.cuda()  
        tokenized_attr = self.tokenized_attr.cuda()
        
        prompt_features = []; logits = []
        for meta_token, imf in zip(meta_tokens, image_features): 
            shifted_prompt = learnable_attr + meta_token
            shifted_prompt = shifted_prompt.unsqueeze(0).expand(self.n_cls, -1, -1)
            attr_prompt = torch.cat([prefix,  shifted_prompt,  suffix], dim = 1) 
            attr_text_feature = self.extractor.encode_texts(embeds=attr_prompt, tokens=tokenized_attr)
            attr_text_feature = attr_text_feature / attr_text_feature.norm(dim=-1, keepdim=True)
            logit = logit_scale * imf @ attr_text_feature.t()
            prompt_features.append(attr_text_feature)
            logits.append(logit)
        logits = torch.stack(logits)
        prompt_features = torch.stack(prompt_features)
        return prompt_features, logits

    def forward(self, image, label=None):    
        # setting 2-3: AdTriplet (constraints4, Meta)
        # get the meta-token and the prompt feature from the original image. (image1)
        image_features_1 = self.extractor.encode_images(images=image)
        meta_token_1 = self.meta_net(image_features_1)
        prompt_features_1, logit_1 = self.prompt_maker(meta_token_1, image_features_1)
        
        if self.training:
            # for only training phase, 
            # get the meta-token and the prompt feature of the random image pair (image2)
            image2, label_2, img2_path = self.get_image_pair(label)     

            image2 = image2.unsqueeze(0).cuda()
            image_features_2 = self.extractor.encode_images(images=image2)
            meta_token_2 = self.meta_net(image_features_2)
            
            # for augmentation, squeeze dimension of the image .
            image = image.squeeze(0)
            image2 = image2.squeeze(0)
            
            # select 2 augmentation types randomly (non-duplicated)
            augment_type_1, augment_type_2 = random.sample(self.augment_types, 2)
            
            distorted_images_1 = self.image_distortion(image, augment_type_1, augment_type_2) 
            distorted_images_2 = self.image_distortion(image2, augment_type_1, augment_type_2) 
            
            distorted_features_1AB = self.extractor.encode_images(distorted_images_1)
            distorted_features_2AB = self.extractor.encode_images(distorted_images_2)

            meta_tokens_1AB = self.meta_net(distorted_features_1AB)
            meta_tokens_2AB = self.meta_net(distorted_features_2AB)

            meta_tokens_1A, meta_tokens_1B, meta_tokens_2A, meta_tokens_2B = meta_tokens_1AB[0], meta_tokens_1AB[1], meta_tokens_2AB[0], meta_tokens_2AB[1]

            # option 2
            delta_meta_1A = meta_tokens_1A - meta_token_1
            delta_meta_1B = meta_tokens_1B - meta_token_1
            delta_meta_2A = meta_tokens_2A - meta_token_2
            delta_meta_2B = meta_tokens_2B - meta_token_2
            
            # option 1
            ad_tri_loss_1, pos_dist_ad_1, neg_dist_ad_1 = self.triplet_loss(delta_meta_1A, delta_meta_2A, delta_meta_1B)   # anchor, positive, negative
            ad_tri_loss_2, pos_dist_ad_2, neg_dist_ad_2 = self.triplet_loss(delta_meta_2B, delta_meta_1B, delta_meta_2A)   # anchor, positive, negative
            ad_tri_loss = ad_tri_loss_1 + ad_tri_loss_2            
            
            # CE Loss(1)
            ce_loss = F.cross_entropy(logit_1, label)
            alpha = 1.0
            beta = 5.0

            total_loss = alpha * ad_tri_loss + beta * ce_loss
            
            ad_distances = [[pos_dist_ad_1, pos_dist_ad_2], [neg_dist_ad_1, neg_dist_ad_2]]
            return total_loss, ce_loss, ad_tri_loss, ad_distances
        
        logits = logit_1

        if not self.cfg.TEST.NO_TEST:
            meta_tokens_list = [meta_token_1]
            delta_meta_list = []
            for augment_type in self.augment_types:
                distorted_images = self.image_distortion.distort_1(image, augment_type)   # [20, 3, 224, 224]
                distorted_features = self.extractor.encode_images(distorted_images)         # [20,3,224,224]>>[20, 512]
                meta_tokens = self.meta_net(distorted_features)                               # [20, 512]
                delta_meta = meta_tokens - meta_token_1   
                meta_tokens_list.append(meta_tokens)
                delta_meta_list.append(delta_meta)
            meta_tokens_list = torch.stack(meta_tokens_list)        # [13, 20, 512]
            delta_meta_list = torch.stack(delta_meta_list)          # [13, 20, 512]
            tokens_list = [meta_tokens_list, delta_meta_list]
            return logits, tokens_list
        
        return logits

# Augmentation Type learner 
@TRAINER_REGISTRY.register()
class APPL2_SIL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.augment_types = ['color_jitter_red', 'color_jitter_green', 'color_jitter_blue', 'grayscale', 'gaussian_noise', 'gaussian_blur','horizontal_flip', 
                            'vertical_flip', 'rotation_90', 'rotation_180', 'rotation_270', 'crop', 'cutout', 'sobel_filter']
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]
     
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            clip_model.float()

        print("Building PARALLEL_AUGLY")
        self.model = DiVE(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.meta_net.parameters():
            param.requires_grad = True
        
        self.model.learnable_attr.requires_grad = True
        
        # Double check
        enabled = set()           
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                
        print(f"Parameters to be updated: {enabled}")
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Parameter numbers: ', params)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("DiVE", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):        
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                total_loss, ce_loss, ad_tri_loss, ad_distances = model(image, label)
                # total_loss, ce_loss, ad_tri_loss, ad_distances, tokens, deltas  = model(image, label)

            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total_loss, ce_loss, ad_tri_loss, ad_distances = model(image, label)
            # total_loss, ce_loss, ad_tri_loss, ad_distances, tokens, deltas  = model(image, label)
            
            self.total_loss = total_loss
            self.ce_loss = ce_loss
            self.ad_tri_loss = ad_tri_loss 

            self.pos_dist_ad_1 = ad_distances[0][0]
            self.pos_dist_ad_2 = ad_distances[0][1]
            self.neg_dist_ad_1 = ad_distances[1][0]
            self.neg_dist_ad_2 = ad_distances[1][1]
            
            # self.tokens = tokens
            # self.deltas = deltas
            
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        loss_summary = {"loss": total_loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # pdb.set_trace()
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
            
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()            
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
    
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)                 
            batch_time.update(time.time() - end)
            losses.update(loss_summary)                                

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:       
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]

                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx                
            
            for name, meter in losses.meters.items():                  
                self.write_scalar("train/" + name, meter.avg, n_iter)  
                
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            self.write_scalar("loss/total_loss", self.total_loss ,n_iter)
            self.write_scalar("loss/ce_loss", self.ce_loss ,n_iter)
            self.write_scalar("loss/ad_tri_loss", self.ad_tri_loss ,n_iter)
            
            self.write_scalar("distances/pos_dist_ad_1", self.pos_dist_ad_1 ,n_iter)
            self.write_scalar("distances/pos_dist_ad_2", self.pos_dist_ad_2 ,n_iter)
            self.write_scalar("distances/neg_dist_ad_1", self.neg_dist_ad_1 ,n_iter)
            self.write_scalar("distances/neg_dist_ad_2", self.neg_dist_ad_2 ,n_iter)
                        
            end = time.time()


    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # self.cfg.TEST.FINAL_MODEL 
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )            
            # pdb.set_trace()
            self.AUG_SCORE()
            self.TSNE()

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # pdb.set_trace()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # pdb.set_trace()
            input, label = self.parse_batch_test(batch)
            if batch_idx == 0:
                output, tokens_list = self.model_inference(input)
                self.tokens_list = tokens_list
            else:
                output, _  = self.model_inference(input)
            self.evaluator.process(output, label)
        
        results = self.evaluator.evaluate()
        # OrderedDict([('accuracy', 75.0), ('error_rate', 25.0), ('macro_f1', 74.3073593073593)])

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def TSNE(self):
        ''' ################# For TSNE-Visualization #################
            self.augment_types = ['color_jitter_red', 'color_jitter_green', 'color_jitter_blue', 'grayscale', 'gaussian_noise', 
            'gaussian_blur','horizontal_flip', 'vertical_flip', 'rotation_90', 'rotation_180', 'rotation_270', 'crop', 'cutout']
        '''
        augment_types = self.augment_types
        os.makedirs(f'{self.output_dir}/tsne_meta', exist_ok=True)
        os.makedirs(f'{self.output_dir}/tsne_delta', exist_ok=True)
        meta_tokens_list, delta_meta_list = self.tokens_list                             # [14, 20, 512], [13, 20, 512]
        val_N = meta_tokens_list.size(1)
        print('######## validation image 개수 :', val_N)
        meta_token_data = meta_tokens_list.cpu().detach().view(-1, 512).numpy()                 # (14,20,512)>>(280,512)
        delta_token_data = delta_meta_list.cpu().detach().view(-1, 512).numpy()                 # (13,20,512)>>(260,512)

        tsne_model = TSNE(n_components=2, perplexity=10, verbose=1, random_state=123)
        tsne_results_1 = tsne_model.fit_transform(meta_token_data)
        tsne_results_2 = tsne_model.fit_transform(delta_token_data)

        colors = cm.rainbow(np.linspace(0, 1, len(augment_types) + 1))

        plt.figure(figsize=(12, 8))
        for i, aug_type in enumerate(['original_image'] + augment_types):
            indices = (i*val_N, (i+1)*val_N)
            plt.scatter(tsne_results_1[indices[0]:indices[1], 0], tsne_results_1[indices[0]:indices[1], 1], color=colors[i], label=aug_type)
        plt.title(f'epoch_{self.epoch+1} Meta_tokens')
        plt.legend()
        plt.savefig(f'{self.output_dir}/tsne_meta/{self.epoch+1}_meta.png')

        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        plt.figure(figsize=(12, 8))
        for i, aug_type in enumerate(augment_types):
            indices = (i*val_N, (i+1)*val_N)
            plt.scatter(tsne_results_2[indices[0]:indices[1], 0], tsne_results_2[indices[0]:indices[1], 1], color=colors[i], label=aug_type)
        plt.title(f'epoch_{self.epoch+1} Delta_meta_tokens')
        plt.legend()
        plt.savefig(f'{self.output_dir}/tsne_delta/{self.epoch+1}_delta.png')
        
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


    def AUG_SCORE(self):

        def silhouette_score_all(delta_meta_list, val_N):
            delta_token_data = delta_meta_list.cpu().detach().view(-1, 512).numpy()                 # (13,valN,512)>>(13*valN,512)
            n_clusters = 14
            cluster_labels = [num for num in range(n_clusters) for _ in range(val_N)]

            silhouette_avg = silhouette_score(delta_token_data, cluster_labels)
            # print(f"전체 데이터에 대한 평균 실루엣 점수: {silhouette_avg}")                
            # 개별 군집의 실루엣 점수 계산
            silhouette_values = silhouette_samples(delta_token_data, cluster_labels)

            cluster_silhouette_avg_list = []
            for i in range(n_clusters):
                # cluster_silhouette_values = silhouette_values[cluster_labels == i]
                start_idx = i * val_N
                end_idx = (i + 1) * val_N
                cluster_silhouette_values = silhouette_values[start_idx: end_idx]
                cluster_silhouette_avg = np.mean(cluster_silhouette_values)
                cluster_silhouette_avg_list.append(cluster_silhouette_avg)
                # print(f"클러스터 {i}의 평균 실루엣 점수: {cluster_silhouette_avg}")
            return cluster_silhouette_avg_list, silhouette_avg
            
        def silhouette_score_two(delta_meta_list, val_N):
            delta_meta_list = delta_meta_list.cpu().detach()
            pair_silhouette_avg_list = []
            for pair in augment_pairs:      # 78
                a1, a2 = pair
                a1_data = delta_meta_list[augment_types.index(a1)]      # [20, 512]
                a2_data = delta_meta_list[augment_types.index(a2)]      # [20, 512]
                delta_token_pair_data = torch.cat((a1_data, a2_data), dim=0)    # [40, 512]
                cluster_labels = [num for num in range(2) for _ in range(val_N)]
                # silhouette_values = silhouette_samples(delta_token_pair_data, cluster_labels)     # 값별 점수 
                silhouette_avg = silhouette_score(delta_token_pair_data, cluster_labels)
                pair_silhouette_avg_list.append(silhouette_avg)
            return pair_silhouette_avg_list
        

        augment_types = self.augment_types
        augment_pairs = list(combinations(augment_types, 2))        # 78
        augment_pairs_str = [pair[0]+'-'+pair[1] for pair in combinations(augment_types, 2)]

        meta_tokens_list, delta_meta_list = self.tokens_list                             # [14, valN, 512], [13, valN, 512]
        val_N = meta_tokens_list.size(1)

        ## 쌍에대한 실루엣 점수
        silhouette_avg_list = silhouette_score_two(delta_meta_list, val_N)
        ## 군집별 점수, 전체 실루엣 점수 
        cluster_silhouette_avg_list, silhouette_avg_all = silhouette_score_all(delta_meta_list, val_N)
        print(f'silhouette_avg_all: ', silhouette_avg_all)
        
        plt.figure(figsize=(12, 6)) 
        os.makedirs(f'{self.output_dir}/silhouette_scores', exist_ok=True)
        plt.bar(augment_types, cluster_silhouette_avg_list, color='skyblue')
        plt.xticks(rotation=45, ha='right')  # x축 레이블을 45도 기울여서 표시합니다.
        plt.title(f'epoch_{self.epoch+1} Delta token Silhouette Scores for Different Augmentation Types')
        plt.xlabel('Augmentation Type') ; plt.ylabel('Silhouette Score')
        for i, score in enumerate(cluster_silhouette_avg_list):
            plt.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom')
        plt.savefig(f'{self.output_dir}/silhouette_scores/delta_{self.epoch+1}.png', dpi=100, bbox_inches='tight')

        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        ### pair별 실루엣점수 bar 그래프 
        plt.figure(figsize=(8, 12)) 
        os.makedirs(f'{self.output_dir}/silhouette_scores', exist_ok=True)
        plt.barh(augment_pairs_str, silhouette_avg_list, color='skyblue')
        plt.yticks(rotation=0, fontsize=7)
        plt.title(f'epoch_{self.epoch+1} Pair Delta token Silhouette Scores for Different Augmentation Types')
        plt.ylabel('Augmentation Type Pair') ; plt.xlabel('Silhouette Score')
        for i, score in enumerate(silhouette_avg_list): 
            plt.text(score + 0.01, i, f'{score:.2f}', ha='center', va='bottom')
        plt.savefig(f'{self.output_dir}/silhouette_scores/pair_delta_{self.epoch+1}.png', dpi=100, bbox_inches='tight')

        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

