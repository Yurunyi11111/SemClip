from ast import mod
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.nn.init as init

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from .clip import ClipPredictor
# from .roi_head import ClipRes5ROIHeadsAttn


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean=[0.48145466, 0.4578275, 0.40821073]
        std=[0.26862954, 0.26130258, 0.27577711] 
        clip_images = [ T.functional.normalize(ci.flip(0)/255, mean,std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i  for i in clip_images])
        return clip_images


    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]#batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        
        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        try:
            _, detector_losses,scores_re = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses, = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self,N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

            
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                logits,proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
             
            try:
                results, _,_ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except:
                results, _,_ = self.roi_heads(images, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)


            return allresults
        else:
            return results


def gate_unit(visual_feature, text_feature, W_v, W_t):
    """
    门控单元
    """
    text_feature_expanded = text_feature.expand(visual_feature.size(0), -1)
    concat_feat = torch.cat([visual_feature, text_feature_expanded], dim=-1)  
    g_v = torch.sigmoid(W_v(concat_feat))  
    g_t = torch.sigmoid(W_t(concat_feat)) 
    return g_v, g_t

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainable(ClipRCNNWithClipBackbone):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)

        domain_text = {'day': 'an image taken during the day'}
        with open('prunedprompts2.txt','r') as f:
            for ind,l in enumerate(f):
                domain_text.update({str(ind):l.strip()})

        import clip
        self.domain_tk = dict([(k,clip.tokenize(t)) for k,t in domain_text.items()])
        self.apply_aug = cfg.AUG_PROB

        self.cos_sim_loss = torch.tensor(0.5, device=self.device, requires_grad=True)

        feature_dim = 1024  # Assuming the feature dimension (e.g., res4 features)
        self.style_params = nn.ModuleDict()
        self.cos_sim_loss_avg = 0.25  
        self.loss_avg_alpha = 0.9    


        clsnames = ['bus' ,'bike', 'car', 'motor', 'person', 'rider' ,'truck']
        self.clip_predictor = ClipPredictor(cfg.MODEL.CLIP_IMAGE_ENCODER_NAME, 2048 , cfg.MODEL.DEVICE,clsnames)


        for domain_name in self.domain_tk.keys():
            if domain_name != 'day':  

                style_mean = nn.Parameter(torch.empty(1,feature_dim, 1, 1))
                style_std = nn.Parameter(torch.empty(1,feature_dim, 1, 1))
                init.normal_(style_mean, mean=0.0, std=0.02)
                init.normal_(style_std, mean=1.0, std=0.02)
                self.style_params[domain_name] = nn.ParameterDict({
                    'style_mean': style_mean,
                    'style_std': style_std
                })





    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]#batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        features_ori = features


        available_domains = [name for name in self.domain_tk.keys() if name != 'day']
        features_aug = None
        if np.random.rand(1) >self.apply_aug:
            oid = np.random.choice(available_domains)
            mu = self.style_params[oid]['style_mean']
            std  = self.style_params[oid]['style_std']
            change = features['res4'] * std + mu
            alpha = 0.1
            features_aug = features
            features['res4'] = alpha* change.to(features['res4'].device) + (1 - alpha)*features['res4'] 
            alpha2 = 1.0
            

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        try:
            _, detector_losses,scores_re = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses,scores_re = self.roi_heads(images, features, proposals, gt_instances, None)
            
        with torch.no_grad():
            if features_aug is not None:
                try:
                    _, _,scores_aug = self.roi_heads(images, features_ori, proposals, gt_instances, None, self.backbone)
                except Exception as e:
                    print(e)
                    _, _,scores_aug = self.roi_heads(images, features_ori, proposals, gt_instances, None)

                cos_sim_loss = self.style_consistency(scores_re, scores_aug)
            
                assert not torch.isnan(cos_sim_loss).any(), "cos_sim_loss contains NaN"
                assert not torch.isinf(cos_sim_loss).any(), "cos_sim_loss contains Inf"
            
                self.cos_sim_loss_avg = self.loss_avg_alpha * self.cos_sim_loss_avg + (1 - self.loss_avg_alpha) * cos_sim_loss.item()
                self.cos_sim_loss_value = self.cos_sim_loss_avg
            else:
                cos_sim_loss = torch.tensor(self.cos_sim_loss_avg, device=self.device, requires_grad=False)


        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if features_aug is not None:
            losses["cos_sim_loss"] = cos_sim_loss*0.1
        return losses
    
    def style_consistency(self, im_prob, aug_prob):
        p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
        consistency_loss = (
            F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
            F.kl_div(p_mixture, im_prob, reduction='batchmean')
        ) / 2.
        consistency_loss = torch.clamp(consistency_loss, min=0.001, max=5.0)
        return consistency_loss
    
    def calc_mean_std(self,feat, eps=1e-5):

        size = feat.shape
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def opt_offsets(self, batched_inputs,iter):
         
        crops_clip = None
        if 'randomcrops' in batched_inputs[0]:
            rcrops = [x['randomcrops'] for x in batched_inputs]
            rcrops = torch.cat(rcrops,0)
            crops_clip = rcrops.flip(1)/255
            mean=[0.48145466, 0.4578275, 0.40821073]
            std=[0.26862954, 0.26130258, 0.27577711]
            crops_clip = T.functional.normalize(crops_clip,mean,std)
            crops_clip = crops_clip.cuda()

        with torch.no_grad():
            features = self.backbone(crops_clip)

        if iter == 0:
            print("iter==0")
            
            feat_mean, feat_std = self.calc_mean_std(features['res4'])

            feat_mean = feat_mean.mean(dim=0, keepdim=True)  
            feat_std = feat_std.mean(dim=0, keepdim=True)    


            for domain_name in self.style_params.keys():
                #print("feat+mean",feat_mean.shape)
                self.style_params[domain_name]['style_mean'].data.copy_(feat_mean)
                self.style_params[domain_name]['style_std'].data.copy_(feat_std)


        losses = {}
        total_dist = 0
        total_reg = 0
        total_chgn = 0 

        device = features['res4'].device
        W_v = nn.Linear( 1024, 512).to(device)
        W_t = nn.Linear( 1024, 512).to(device)


        for i,val in enumerate(self.domain_tk.items()):
            name , dtk = val
            if name == 'day':
                continue
            

            
            with torch.no_grad():
                wo_aug_im_embed = self.backbone.attention_global_pool(self.backbone.forward_res5(features['res4']))
                wo_aug_im_embed  = wo_aug_im_embed/wo_aug_im_embed.norm(dim=-1,keepdim=True)
                
                day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(self.domain_tk['day'].cuda())
                day_text_embed = day_text_embed/day_text_embed.norm(dim=-1,keepdim=True)
                new_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(dtk.cuda())
                new_text_embed = new_text_embed/new_text_embed.norm(dim=-1,keepdim=True)
                
                text_off = (new_text_embed - day_text_embed)
                text_off = text_off/text_off.norm(dim=-1,keepdim=True)

                g_v, g_t = gate_unit(wo_aug_im_embed, text_off, W_v, W_t)

                wo_aug_im_tsl = g_v * wo_aug_im_embed + g_t * text_off
                wo_aug_im_tsl = wo_aug_im_tsl/wo_aug_im_tsl.norm(dim=-1,keepdim=True)
                wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0,2,1)

            aug_feat1 = features['res4'] * self.style_params[name]['style_std'] + self.style_params[name]['style_mean']

            
            alpha = 0.1
            aug_feat = alpha* aug_feat1.to(features['res4'].device) + (1 - alpha)*features['res4']

            x = self.backbone.forward_res5(aug_feat)
            im_embed = self.backbone.attention_global_pool(x)
            im_embed = im_embed/im_embed.norm(dim=-1,keepdim=True)

            cos_dist = 1 - im_embed.unsqueeze(1).bmm(wo_aug_im_tsl)
            dist_loss = cos_dist.mean()
            l1loss = torch.nn.functional.l1_loss(im_embed, wo_aug_im_embed)

            total_dist += dist_loss
            total_reg += l1loss

        losses.update({f'cos_dist_loss_{name}': total_dist/len(self.domain_tk), f'reg_loss_{name}': total_reg/len(self.domain_tk)})
        
        
        return losses
