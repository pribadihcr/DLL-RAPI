# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeDet_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  mc.IMAGE_WIDTH           = 480#1248
  mc.IMAGE_HEIGHT          = 240#384
  mc.BATCH_SIZE            = 20

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_LANDMARK = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0
  mc.LOSS_COEF_POSE        = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.1#0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = False
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9
  mc.POINTS                = 10

  return mc

def set_anchors(mc):
  H, W, B = 15, 30, 9  # 24, 78, 9
  anchor_shapes = np.reshape(
        [np.array(
            [[mc.IMAGE_HEIGHT / (384 / 36.), mc.IMAGE_WIDTH / (1248 / 37.)], [mc.IMAGE_HEIGHT / (384 / 366.), mc.IMAGE_WIDTH / (1248 / 174.)],
             [mc.IMAGE_HEIGHT / (384 / 115.), mc.IMAGE_WIDTH / (1248 / 59.)],
             [mc.IMAGE_HEIGHT / (384 / 162.), mc.IMAGE_WIDTH / (1248 / 87.)], [mc.IMAGE_HEIGHT / (384 / 38.), mc.IMAGE_WIDTH / (1248 / 90.)],
             [mc.IMAGE_HEIGHT / (384 / 258.), mc.IMAGE_WIDTH / (1248 / 173.)],
             [mc.IMAGE_HEIGHT / (384 / 224.), mc.IMAGE_WIDTH / (1248 / 108.)], [mc.IMAGE_HEIGHT / (384 / 78.), mc.IMAGE_WIDTH / (1248 / 170.)],
             [mc.IMAGE_HEIGHT / (384 / 72.), mc.IMAGE_WIDTH / (1248 / 43.)]])] * H * W,
        (H, W, B, 2)
  )
  #H, W, B = 24, 78, 9
  #anchor_shapes = np.reshape(
  #    [np.array(
  #        [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
  #         [ 162.,  87.], [  38.,  90.], [ 258., 173.],
  #         [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
  #    (H, W, B, 2)
  #)
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
