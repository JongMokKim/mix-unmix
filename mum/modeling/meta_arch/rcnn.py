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
    # def __init__(self,cfg):
    #     super(TwoStagePseudoLabGeneralizedRCNN, self).__init__()
    #
    #     pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    #     pooler_scales     = [1/4,1/8,1/16,1/32,1/64]  # freeze stride config
    #     sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    #     pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
    #
    #     self.roi_pooler = ROIPooler(
    #         output_size=pooler_resolution,
    #         scales=pooler_scales,
    #         sampling_ratio=sampling_ratio,
    #         pooler_type=pooler_type,
    #         )
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False,
            ts=None, nt=None, tile_mask=None, features_unsup_k=None, proposals_unsup_k=None,
            roi_pooler=None, f_list=None, lambda_feat = None, gi_type=None, lambda_rel=None,
            tile_prop=1.
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        if branch == 'supervised_tile_reassemble_gi':

            images = self.preprocess_image(batched_inputs)

            bs, c, h, w = images.tensor.shape

            mask = torch.argsort(torch.rand(bs // nt, nt, ts, ts), dim=1).cuda()
            inv_mask = torch.argsort(mask, dim=1).cuda()

            img_mask = mask.view(bs // nt, nt, 1, ts, ts)
            img_mask = img_mask.repeat_interleave(3, dim=2)
            img_mask = img_mask.repeat_interleave(h // ts, dim=3)
            img_mask = img_mask.repeat_interleave(w // ts, dim=4)

            img_tiled = images.tensor.view(bs // nt, nt, c, h, w)
            img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
            img_tiled = img_tiled.view(bs, c, h, w)

            # #####################################################################
            # ## for debug
            # pixel_mean = torch.tensor((103.53, 116.28, 123.675)).view(1,3,1,1).cuda()
            #
            # imgs = images.tensor + pixel_mean
            # imgs = imgs.permute(0,2,3,1)
            # for ii,img in enumerate(imgs):
            #     img = img.data.cpu().numpy().astype(np.uint8)
            #     cv2.imwrite('data_sanity/orig_img_{}.png'.format(ii),img)
            #
            # imgs = img_tiled + pixel_mean
            # imgs = imgs.permute(0,2,3,1)
            # for ii,img in enumerate(imgs):
            #     img = img.data.cpu().numpy().astype(np.uint8)
            #     cv2.imwrite('data_sanity/tiled_img_{}.png'.format(ii),img)
            #
            # raise RuntimeError('End data sanity')
            # ######################################################################

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            # get features
            features = self.backbone(img_tiled)

            # reassemble features
            for pn, feat in features.items():
                bs, c, h, w = feat.shape

                h_ = h//ts * ts
                w_ = w//ts * ts

                if h != h_ or w != w_:
                    feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')

                feat_tiled = feat.view(bs//nt,nt,c,h_,w_)
                feat_mask = inv_mask.view(bs//nt,nt,1,ts,ts)
                feat_mask = feat_mask.repeat_interleave(c,dim=2)
                feat_mask = feat_mask.repeat_interleave(h_//ts, dim=3)
                feat_mask = feat_mask.repeat_interleave(w_//ts, dim=4)

                feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
                feat_tiled = feat_tiled.view(bs,c,h_,w_)

                if h != h_ or w != w_:
                    feat_tiled = torch.nn.functional.interpolate(feat_tiled, size=(h, w), mode='bilinear')

                features[pn] = feat_tiled

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # GI Loss
            gi_loss = self.calculate_gi_loss(features_unsup_k, features, proposals_unsup_k, proposals_rpn,
                                             roi_pooler, f_list, lambda_feat, gi_type, lambda_rel)

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(gi_loss)
            return losses, [], [], None

        elif branch == 'supervised_tile_reassemble':

            images = self.preprocess_image(batched_inputs)

            bs, c, h, w = images.tensor.shape
            p = random.random()

            if p < tile_prop:
                mask = torch.argsort(torch.rand(bs // nt, nt, ts, ts), dim=1).cuda()
                inv_mask = torch.argsort(mask, dim=1).cuda()

                img_mask = mask.view(bs // nt, nt, 1, ts, ts)
                img_mask = img_mask.repeat_interleave(3, dim=2)
                img_mask = img_mask.repeat_interleave(h // ts, dim=3)
                img_mask = img_mask.repeat_interleave(w // ts, dim=4)

                img_tiled = images.tensor.view(bs // nt, nt, c, h, w)
                img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
                img_tiled = img_tiled.view(bs, c, h, w)
            else: # non-tiling
                img_tiled = images.tensor

            # #####################################################################
            # ## for debug
            # pixel_mean = torch.tensor((103.53, 116.28, 123.675)).view(1,3,1,1).cuda()
            #
            # imgs = images.tensor + pixel_mean
            # imgs = imgs.permute(0,2,3,1)
            # for ii,img in enumerate(imgs):
            #     img = img.data.cpu().numpy().astype(np.uint8)
            #     cv2.imwrite('data_sanity/orig_img_{}.png'.format(ii),img)
            #
            # imgs = img_tiled + pixel_mean
            # imgs = imgs.permute(0,2,3,1)
            # for ii,img in enumerate(imgs):
            #     img = img.data.cpu().numpy().astype(np.uint8)
            #     cv2.imwrite('data_sanity/tiled_img_{}.png'.format(ii),img)
            #
            # raise RuntimeError('End data sanity')
            # ######################################################################

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            # get features
            features = self.backbone(img_tiled)

            # reassemble features
            if p < tile_prop:
                for pn, feat in features.items():
                    bs, c, h, w = feat.shape

                    h_ = h//ts * ts
                    w_ = w//ts * ts

                    if h_ == 0:
                        h_ = ts
                    if w_ == 0:
                        w_ = ts

                    if h != h_ or w != w_:
                        feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')
                        # print(bs, c, h, w, h_, w_, feat.shape)
                    feat_tiled = feat.view(bs//nt,nt,c,h_,w_)
                    feat_mask = inv_mask.view(bs//nt,nt,1,ts,ts)
                    feat_mask = feat_mask.repeat_interleave(c,dim=2)
                    feat_mask = feat_mask.repeat_interleave(h_//ts, dim=3)
                    feat_mask = feat_mask.repeat_interleave(w_//ts, dim=4)

                    feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
                    feat_tiled = feat_tiled.view(bs,c,h_,w_)

                    if h != h_ or w != w_:
                        feat_tiled = torch.nn.functional.interpolate(feat_tiled, size=(h, w), mode='bilinear')

                    features[pn] = feat_tiled

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

            # if branch == 'unsup_data_weak':
            #     #####################################################################
            #     ## for debug
            #     pixel_mean = torch.tensor((103.53, 116.28, 123.675)).view(1,3,1,1).cuda()
            #
            #     imgs = images.tensor + pixel_mean
            #     imgs = imgs.permute(0,2,3,1)
            #     for ii,img in enumerate(imgs):
            #         img = img.data.cpu().numpy().astype(np.uint8)
            #         cv2.imwrite('data_sanity/weak_img_{}.png'.format(ii),img)
            #     ######################################################################

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            if tile_mask is not None:
                bs, c, h, w = images.tensor.shape

                img_mask = tile_mask.view(bs // nt, nt, 1, ts, ts)
                img_mask = img_mask.repeat_interleave(3, dim=2)
                img_mask = img_mask.repeat_interleave(h // ts, dim=3)
                img_mask = img_mask.repeat_interleave(w // ts, dim=4)

                img_tiled = images.tensor.view(bs // nt, nt, c, h, w)
                img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
                images.tensor = img_tiled.view(bs, c, h, w)

            features = self.backbone(images.tensor)

            if branch == "supervised":

                if tile_mask is not None:
                    inv_mask = torch.argsort(tile_mask, dim=1).cuda()
                    for pn, feat in features.items():
                        bs, c, h, w = feat.shape

                        h_ = h // ts * ts
                        w_ = w // ts * ts

                        if h_ == 0:
                            h_ = ts
                        if w_ == 0:
                            w_ = ts

                        if h != h_ or w != w_:
                            feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')
                            # print(bs, c, h, w, h_, w_, feat.shape)
                        feat_tiled = feat.view(bs // nt, nt, c, h_, w_)
                        feat_mask = inv_mask.view(bs // nt, nt, 1, ts, ts)
                        feat_mask = feat_mask.repeat_interleave(c, dim=2)
                        feat_mask = feat_mask.repeat_interleave(h_ // ts, dim=3)
                        feat_mask = feat_mask.repeat_interleave(w_ // ts, dim=4)

                        feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
                        feat_tiled = feat_tiled.view(bs, c, h_, w_)

                        if h != h_ or w != w_:
                            feat_tiled = torch.nn.functional.interpolate(feat_tiled, size=(h, w), mode='bilinear')

                        features[pn] = feat_tiled

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


    def calculate_gi_loss(self, feat_k, feat_q, proposal_k, proposal_q, roi_pooler, f_list, lambda_feat, gi_type, lambda_rel):
        '''
        Calculate General Instance losses
        '''

        if gi_type=='wholefeat':
            feat_loss = 0
            for f in f_list:
                f_k = feat_k[f]
                f_q = feat_q[f]
                try:
                    tmp = ((f_k-f_q)**2).sum(1).sqrt()
                except:
                    f_k = torch.nn.functional.interpolate(f_k, (f_q.shape[2:]))
                    tmp = ((f_k - f_q) ** 2).sum(1).sqrt()
                feat_loss += tmp.mean()

            feat_loss = feat_loss * lambda_feat / len(f_list)

            # raise RuntimeError('GI DEBUG END')
            return {'feat_loss':feat_loss}

        else:
            # 1. gen GI instance
            gi_list = []
            for prop_k, prop_q in zip(proposal_k, proposal_q):

                k_idx = prop_k.objectness_logits > 0.
                q_idx = prop_q.objectness_logits > 0.

                k_boxes = prop_k.proposal_boxes.tensor[k_idx]
                q_boxes = prop_q.proposal_boxes.tensor[q_idx]

                # print(k_boxes.shape, q_boxes.shape)
                merge_boxes = torch.cat([k_boxes,q_boxes], dim=0)
                fake_scores_k = torch.ones_like(k_boxes[:,0]) * 1.
                fake_scores_q = torch.ones_like(q_boxes[:,0]) * 0.5
                merge_fake_scores = torch.cat([fake_scores_k, fake_scores_q], dim=0)
                dom_k = torchvision.ops.boxes.nms(merge_boxes,merge_fake_scores,iou_threshold=0.7)

                fake_scores_k = torch.ones_like(k_boxes[:,0]) * 0.5
                fake_scores_q = torch.ones_like(q_boxes[:,0]) * 1.
                merge_fake_scores = torch.cat([fake_scores_k, fake_scores_q], dim=0)
                dom_q = torchvision.ops.boxes.nms(merge_boxes,merge_fake_scores,iou_threshold=0.7)

                # find intersection of dom_k, dom_q
                uniq, counts = torch.cat([dom_k,dom_q]).unique(return_counts=True)
                diff = uniq[counts == 1]

                gi_list.append(merge_boxes[diff])
                # print(merge_boxes[diff].shape)

            # 2. calculate GI loss
            feat_k_list = [feat_k[f] for f in f_list]
            feat_q_list = [feat_q[f] for f in f_list]

            # print(len(gi_list))
            # print(gi_list[0].shape)
            gi_list = [Boxes(e) for e in gi_list]

            if gi_type == 'gifeat':

                feat_roi_k = roi_pooler(feat_k_list, gi_list)
                feat_roi_q = roi_pooler(feat_q_list, gi_list)

                # feat_loss = torch.cdist(feat_roi_k, feat_roi_q, p=2.0)
                # print(feat_loss.shape, feat_loss.mean(), feat_loss[0,:,0,0])

                feat_loss = ((feat_roi_k-feat_roi_q)**2).sum(1).sqrt()
                feat_loss = feat_loss.mean() * lambda_feat

                # raise RuntimeError('GI DEBUG END')
                return {'feat_loss':feat_loss}

            elif gi_type == 'gifeatrel':

                feat_loss = 0
                rel_loss = 0

                accum_batch = 0
                for b in range(len(gi_list)):
                    gi = [gi_list[b]]

                    if gi[0].tensor.shape[0] > 0:
                        accum_batch +=1
                        f_k = [e[b:b+1] for e in feat_k_list]
                        f_q = [e[b:b + 1] for e in feat_q_list]

                        feat_roi_k = roi_pooler(f_k, gi)
                        feat_roi_q = roi_pooler(f_q, gi)

                        tmp = ((feat_roi_k-feat_roi_q)**2).sum(1).sqrt()

                        feat_loss += tmp.mean()

                        feat_roi_k = torch.mean(feat_roi_k, dim=(2,3))
                        feat_roi_q = torch.mean(feat_roi_q, dim=(2, 3))

                        feat_roi_k /= torch.norm(feat_roi_k, dim=1, keepdim=True)
                        feat_roi_q /= torch.norm(feat_roi_q, dim=1, keepdim=True)

                        cos_k = torch.matmul(feat_roi_k, feat_roi_k.t())
                        cos_q = torch.matmul(feat_roi_q, feat_roi_q.t())

                        rel_loss += ((cos_k-cos_q)**2).sqrt().mean()

                        if torch.isnan(feat_loss):
                            print('====================')
                            print('nan occur!!')
                            print('after_normk', feat_roi_k.mean())
                            print('after_normq', feat_roi_q.mean())
                            print('cos', cos_k.mean(), cos_k.shape)
                            raise RuntimeError('error end')

                feat_loss = feat_loss * lambda_feat / accum_batch
                rel_loss = rel_loss * lambda_rel / accum_batch

                return {'feat_loss':feat_loss, 'rel_loss':rel_loss}


    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results


@META_ARCH_REGISTRY.register()
class TUTGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, TUT=False, ts=4, tut_layer=None, tutprob=0.
    ):
        if (not self.training):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if TUT:
            bs, c, h, w = images.tensor.shape
            nt=4
            tile_mask = torch.argsort(
                torch.rand(bs // 4, 4, ts, ts), dim=1).cuda()

            img_mask = tile_mask.view(bs // nt, nt, 1, ts, ts)
            img_mask = img_mask.repeat_interleave(3, dim=2)
            img_mask = img_mask.repeat_interleave(h // ts, dim=3)
            img_mask = img_mask.repeat_interleave(w // ts, dim=4)

            img_tiled = images.tensor.view(bs // nt, nt, c, h, w)
            img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
            img_tiled = img_tiled.view(bs, c, h, w)
        else:
            tile_mask = None
            img_tiled = None

        if img_tiled is None:
            features = self.backbone(images.tensor)
        else:
            features = self.backbone(img_tiled, mask=tile_mask)

        if tile_mask is not None and tut_layer is None:
            inv_mask = torch.argsort(tile_mask, dim=1).cuda()
            nt=4
            for pn, feat in features.items():
                bs, c, h, w = feat.shape

                h_ = h // ts * ts
                w_ = w // ts * ts

                if h_ == 0:
                    h_ = ts
                if w_ == 0:
                    w_ = ts

                if h != h_ or w != w_:
                    feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')
                    # print(bs, c, h, w, h_, w_, feat.shape)
                feat_tiled = feat.view(bs // nt, nt, c, h_, w_)
                feat_mask = inv_mask.view(bs // nt, nt, 1, ts, ts)
                feat_mask = feat_mask.repeat_interleave(c, dim=2)
                feat_mask = feat_mask.repeat_interleave(h_ // ts, dim=3)
                feat_mask = feat_mask.repeat_interleave(w_ // ts, dim=4)

                feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
                feat_tiled = feat_tiled.view(bs, c, h_, w_)

                if h != h_ or w != w_:
                    feat_tiled = torch.nn.functional.interpolate(feat_tiled, size=(h, w), mode='bilinear')

                features[pn] = feat_tiled

        # Region proposal network
        proposals_rpn, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )

        # # roi_head lower branch
        _, detector_losses = self.roi_heads(
            images, features, proposals_rpn, gt_instances#, branch=branch
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


