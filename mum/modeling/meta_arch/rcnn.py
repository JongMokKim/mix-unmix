# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch
import cv2
import numpy as np
from detectron2.modeling.poolers import ROIPooler
import torchvision
from detectron2.structures.boxes import Boxes
import random

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    def forward(
        self, batched_inputs, branch="supervised",  val_mode=False,
            nt=None, ng=None, mix_mask=None, tile_prop=1.):

        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        if branch == 'supervised_mix_unmix':

            images = self.preprocess_image(batched_inputs)

            bs, c, h, w = images.tensor.shape
            p = random.random()

            # generate mixing mask and mix image tiles
            if p < tile_prop:
                mix_mask = torch.argsort(torch.rand(bs // ng, ng, nt, nt), dim=1).cuda()
                inv_mask = torch.argsort(mix_mask, dim=1).cuda()

                img_mask = mix_mask.view(bs // ng, ng, 1, nt, nt)
                img_mask = img_mask.repeat_interleave(3, dim=2)
                img_mask = img_mask.repeat_interleave(h // nt, dim=3)
                img_mask = img_mask.repeat_interleave(w // nt, dim=4)

                img_mixed = images.tensor.view(bs // ng, ng, c, h, w)
                img_mixed = torch.gather(img_mixed, dim=1, index=img_mask)
                img_mixed = img_mixed.view(bs, c, h, w)
            else: # non-mixing
                img_mixed = images.tensor

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            # get features
            features = self.backbone(img_mixed)

            # unmix features
            if p < tile_prop:
                for pn, feat in features.items():
                    bs, c, h, w = feat.shape

                    h_ = h//nt * nt
                    w_ = w//nt * nt

                    if h_ == 0:
                        h_ = nt
                    if w_ == 0:
                        w_ = nt

                    if h != h_ or w != w_:
                        feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')

                    feat_mixed = feat.view(bs//ng,ng,c,h_,w_)
                    feat_mask = inv_mask.view(bs//ng,ng,1,nt,nt)
                    feat_mask = feat_mask.repeat_interleave(c,dim=2)
                    feat_mask = feat_mask.repeat_interleave(h_//nt, dim=3)
                    feat_mask = feat_mask.repeat_interleave(w_//nt, dim=4)

                    feat_mixed = torch.gather(feat_mixed, dim=1, index=feat_mask)
                    feat_mixed = feat_mixed.view(bs,c,h_,w_)

                    if h != h_ or w != w_:
                        feat_mixed = torch.nn.functional.interpolate(feat_mixed, size=(h, w), mode='bilinear')

                    features[pn] = feat_mixed

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        else:
            images = self.preprocess_image(batched_inputs)

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            if mix_mask is not None:
                bs, c, h, w = images.tensor.shape

                img_mask = mix_mask.view(bs // ng, ng, 1, nt, nt)
                img_mask = img_mask.repeat_interleave(3, dim=2)
                img_mask = img_mask.repeat_interleave(h // nt, dim=3)
                img_mask = img_mask.repeat_interleave(w // nt, dim=4)

                img_mixed = images.tensor.view(bs // ng, ng, c, h, w)
                img_mixed = torch.gather(img_mixed, dim=1, index=img_mask)
                images.tensor = img_mixed.view(bs, c, h, w)

            features = self.backbone(images.tensor)

            if branch == "supervised":
                if mix_mask is not None:
                    inv_mask = torch.argsort(mix_mask, dim=1).cuda()
                    for pn, feat in features.items():
                        bs, c, h, w = feat.shape

                        h_ = h // nt * nt
                        w_ = w // nt * nt

                        if h_ == 0:
                            h_ = nt
                        if w_ == 0:
                            w_ = nt

                        if h != h_ or w != w_:
                            feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')
                            # print(bs, c, h, w, h_, w_, feat.shape)
                        feat_mixed = feat.view(bs // ng, ng, c, h_, w_)
                        feat_mask = inv_mask.view(bs // ng, ng, 1, nt, nt)
                        feat_mask = feat_mask.repeat_interleave(c, dim=2)
                        feat_mask = feat_mask.repeat_interleave(h_ // nt, dim=3)
                        feat_mask = feat_mask.repeat_interleave(w_ // nt, dim=4)

                        feat_mixed = torch.gather(feat_mixed, dim=1, index=feat_mask)
                        feat_mixed = feat_mixed.view(bs, c, h_, w_)

                        if h != h_ or w != w_:
                            feat_mixed = torch.nn.functional.interpolate(feat_mixed, size=(h, w), mode='bilinear')

                        features[pn] = feat_mixed

                # Region proposal network
                proposals_rpn, proposal_losses = self.proposal_generator(
                    images, features, gt_instances
                )

                # # roi_head lower branch
                _, detector_losses = self.roi_heads(
                    images, features, proposals_rpn, gt_instances, branch=branch
                )

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                return losses, [], [], None

            elif branch == "unsup_data_weak":

                # Region proposal network
                proposals_rpn, _ = self.proposal_generator(
                    images, features, None, compute_loss=False
                )

                # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )

                return features, proposals_rpn, proposals_roih, ROI_predictions

            elif branch == "val_loss":

                # Region proposal network
                proposals_rpn, proposal_losses = self.proposal_generator(
                    images, features, gt_instances, compute_val_loss=True
                )

                # roi_head lower branch
                _, detector_losses = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    gt_instances,
                    branch=branch,
                    compute_val_loss=True,
                )

                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                return losses, [], [], None




