# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    # Initialize SWIN node
    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.OUT_FEATURES = []

    _C.MODEL.BACKBONE.TUT_LAYER = None

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ubteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.Suponly=False
    _C.SEMISUPNET.Augtype = "strong"
    _C.SEMISUPNET.TUTprob = 0.

    ## My Custom
    _C.SEMISUPNET.TYPE = "base" # tile, base,,,
    _C.SEMISUPNET.NT = 4
    _C.SEMISUPNET.NG = 4
    _C.SEMISUPNET.LAMBDA_FEAT = 1e-2
    _C.SEMISUPNET.LAMBDA_REL = 1e-2
    _C.SEMISUPNET.GI_TYPE = "gifeat"
    _C.SEMISUPNET.TILE_PROP = 1.
    _C.SEMISUPNET.SUPTILE = False
    _C.SEMISUPNET.SUPTILE_PROP = 1.
    _C.SEMISUPNET.INIT_MOMENTUM = False
    _C.SEMISUPNET.DYNAMIC_EMA = 0

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

