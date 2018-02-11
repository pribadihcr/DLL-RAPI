# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from PIL import Image
from config import *
from train import _draw_box, _draw_point
from utils.util import bbox_transform
from sklearn.metrics import confusion_matrix, accuracy_score

from nets import *
from h264_decoder_module.h264module import *
from scipy import misc
from tensorflow.python.client import timeline
import json

os.environ['CUDA_VISIBLE_DEVICES'] = ' '


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'video', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    #'checkpoint', '/media/aoi-dl/2680012A69064889/AOITEK/logs/faceclassifier/log_zoo/train191017/model.ckpt-734500',
    #'checkpoint', '/media/aoi-dl/2680012A69064889/AOITEK/logs/faceclassifier/BabyFaceDet/squeezenet_extraDepthwise/model.ckpt-999999',
    'checkpoint', '/media/aoi-dl/2680012A69064889/AOITEK/logs/faceclassifier/BabyFaceDet/squeezenet_extraSE/model.ckpt-246500',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', '/media/aoi-dl/2680012A69064889/AOITEK/logs/faceclassifier/BabyFaceDet/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def evaluation(gt, pred):
    allcm = confusion_matrix(gt, pred)
    allacc = accuracy_score(gt, pred)
    return allcm, allacc

def eval_iou(box1, box2):
    # a small value used to prevent numerical instability
    EPSILON = 1e-16
    xmin = np.maximum(box1[0], box2[0])
    ymin = np.maximum(box1[1], box2[1])
    xmax = np.minimum(box1[2], box2[2])
    ymax = np.minimum(box1[3], box2[3])

    w = np.maximum(0.0, xmax-xmin)
    h = np.maximum(0.0, ymax-ymin)
    intersection = np.multiply(w, h)

    w1 = np.subtract(box1[2], box1[0])
    h1 = np.subtract(box1[3], box1[1])
    w2 = np.subtract(box2[2], box2[0])
    h2 = np.subtract(box2[3], box2[1])

    union = w1*h1 + w2*h2 - intersection

    return intersection/(union+EPSILON)

def draw_box(im, box_list, label_list, pose_list, age_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label, pose, age in zip(box_list, label_list, pose_list, age_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]
    rescale_x = 1.0 * 1280 / 480
    rescale_y = 1.0 * 720 / 240

    xmin *= rescale_x
    xmax *= rescale_x
    ymin *= rescale_y
    ymax *= rescale_y

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label+"; "+pose+"; "+age, (int(xmin), int(ymax)), font, 0.3, c, 1)

def draw_point(im, point_list, color=(0,255,0)):
    #print(point_list)
    rescale_x = 1.0 * 1280 / 480
    rescale_y = 1.0 * 720 / 240
    for points in point_list:
        for p in range(0, 5):
            x = int(points[p]*rescale_x)
            y = int(points[p+5]*rescale_y)
            cv2.circle(im, (x, y), 1, color, 2)


def video_demo():
  """Detect videos."""
  #test_path = '/media/aoi-dl/2680012A69064889/AOITEK/dataset/faceclassifier/BabyFaceDet/testing/'
  test_path = '/home/aoi-dl/SHARE/faceclassifier/BabyFaceDet/src/data/'
  #cap = cv2.VideoCapture(FLAGS.input_path)
  h264_VIDEO_FILE = test_path + 'golden_multitask.mp4'#+'golden_multitask.mp4'  # '/media/rudy/New Volume/babyDL/FACE_Det_Cla_Rec_0.1/database/video/golden_test.h264'
  DEFAULT_H = 720
  DEFAULT_W = 1280
  DEFAULT_C = 3

  pred_emo = []
  pred_pose = []
  pred_age = []

  filename = os.path.basename(h264_VIDEO_FILE)

  print(filename)
  decoder = init_h264dec(h264_VIDEO_FILE)


  # Define the codec and create VideoWriter object
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
  # in_file_name = os.path.split(FLAGS.input_path)[1]
  # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+in_file_name)
  # out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (375,1242), True)
  # out = VideoWriter(out_file_name, frameSize=(1242, 375))
  # out.open()

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    # model_extra_params = [v for v in tf.global_variables() if 'ExtraNet' in v.name and 'moving' not in v.name]
    #
    # params_needinit = [v for v in tf.global_variables() if 'moving' in v.name]
    # print(params_needinit)

    # saver = tf.train.Saver(vs)
    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      # init = tf.variables_initializer(params_needinit)
      # sess.run(init)
      saver.restore(sess, FLAGS.checkpoint)

      #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      #run_metadata = tf.RunMetadata()
      #many_runs_timeline = TimeLiner()

      times = {}
      count = 0
      for i in range(20000):
        t_start = time.time()
        count += 1

        frame = get_frame(decoder, width=DEFAULT_W, height=DEFAULT_H, channels=DEFAULT_C)
        frame_ori = np.copy(frame)
        # print "get frame:", elapsed_time
        if (frame.shape[0]) != DEFAULT_H:
            break
        frame1 = Image.fromarray(frame, mode='RGB')

        img = cv2.cvtColor(np.asarray(frame1), cv2.COLOR_RGB2BGR)
        im = img.astype(np.float32, copy=False)
        img = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        img = img - mc.BGR_MEANS

        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')
        out_im_name1 = os.path.join(FLAGS.out_dir, 'ori_'+str(count).zfill(6) + '.jpg')

        t_reshape = time.time()
        times['get_frame']= t_reshape - t_start

        # Detect
        det_boxes, det_probs, det_class, det_pose, det_age, det_landmarks = sess.run(
            [model.det_boxes, model.det_probs, model.det_class, model.det_pose, model.det_age, model.det_landmarks],
            feed_dict={model.image_input:[img]})#, options=options, run_metadata=run_metadata)

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_landmarks, final_probs, final_pose, final_age, final_class = model.filter_prediction(
            det_boxes[0], det_landmarks[0], det_probs[0],det_pose[0], det_age[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_landmarks = [final_landmarks[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]
        final_pose = [final_pose[idx] for idx in keep_idx]
        final_age = [final_age[idx] for idx in keep_idx]

        # dtpath = '/media/aoi-dl/2680012A69064889/AOITEK/dataset/faceclassifier/BabyFaceDet/testing/'
        # fgt = open(dtpath + 'frame_' + str(i) + '.txt', 'r')
        #
        # bb_gt = []
        # #make sure only get one bounding-box
        # with fgt as f:
        #     for line in f:
        #         #print(line.rstrip())
        #         bbs = line.rstrip().split(' ')
        #         for bb in bbs:
        #             bb_gt.append(int(bb))
        #         break
        #
        # for bbs in final_boxes:
        #     if len(bb_gt) > 0:
        #         print(bbox_transform(bbs), bb_gt)
        #         iou = eval_iou(bbox_transform(bbs), bb_gt)
        #         mean_iou += iou
        #         tot += 1
        #         print("iou",iou)


        times['total']= time.time() - t_start

        t_filter = time.time()
        times['filter'] = t_filter - t_detect

        # Draw boxes

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            '01cry': (255, 191, 0),
            '02cute': (0, 191, 255),
            '03normal': (255, 0, 191),
            '04sleep': (191, 0, 255),
            '05wakeup': (191, 255, 0)
        }

        max_prob = 0.0
        lbl_emo = -1
        lbl_pose = -1
        lbl_age = -1

        for idx, idx1, idx2, prob in zip(final_class, final_pose, final_age, final_probs):
            if prob > max_prob:
                max_prob = prob
                lbl_emo = idx
                lbl_pose = idx1
                lbl_age = idx2
                if idx2 == 1:
                    lbl_emo = 5

        pred_emo.append(lbl_emo)
        pred_pose.append(lbl_pose)
        pred_age.append(lbl_age)

        draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
             for idx, prob in zip(final_class, final_probs)], [mc.POSE_NAMES[idx] for idx in final_pose],
            [mc.AGE_NAMES[idx] for idx in final_age],
            cdict=cls2clr,
        )

        draw_point(frame, final_landmarks, (0, 0, 255))

        t_draw = time.time()
        times['draw'] = t_draw - t_filter

        cv2.imwrite(out_im_name, frame)
        cv2.imwrite(out_im_name1, frame_ori)
        # out.write(frame)

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        print (time_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        #chrome_trace = fetched_timeline.generate_chrome_trace_format()
        #many_runs_timeline.update_timeline(chrome_trace)
      #many_runs_timeline.save('timeline_merged_%d_runs.json' % 1)

  # Release everything if job is finished
  close_h264dec(decoder)
  # out.release()
  cv2.destroyAllWindows()

  # print('mean_iou:', mean_iou/tot)
  with open(test_path + 'gt_emo') as f:
      gt_emo = []
      for line in f:
          lbl = int(line[:-1])
          if lbl==-1:
              lbl = 5
          gt_emo.append(lbl)
  with open(test_path +'gt_age') as f:
      gt_age = []
      for line in f:
          gt_age.append(int(line[:-1]))
  with open(test_path + 'gt_pose') as f:
      gt_pose = []
      for line in f:
          lbl = int(line[:-1])
          # if lbl == 0:
          #     lbl = 1
          # else:
          #     lbl = 0
          gt_pose.append(lbl)

  allcm, allacc = evaluation(gt_emo, pred_emo)
  print(allcm)
  print(allacc)
  allcm, allacc = evaluation(gt_age, pred_age)
  print(allcm)
  print(allacc)
  allcm, allacc = evaluation(gt_pose, pred_pose)
  print(allcm)
  print(allacc)

  np.savetxt('pred.txt', pred_emo)


def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)


      fimgs = glob.glob(os.path.join(FLAGS.input_path,'*.jpg'))	
      for f in fimgs:#glob.iglob(FLAGS.input_path):
        print(f)
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        ss = time.time()
        det_boxes, det_probs, det_class, det_pose, det_landmarks = sess.run(
            [model.det_boxes, model.det_probs, model.det_class, model.det_pose, model.det_landmarks],
            feed_dict={model.image_input:[input_image]})

        # Filter
        final_boxes, final_landmarks, final_probs, final_pose, final_class = model.filter_prediction(
            det_boxes[0], det_landmarks[0], det_probs[0], det_pose[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_landmarks = [final_landmarks[idx] for idx in keep_idx]
        final_pose = [final_pose[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            '01cry': (255, 191, 0),
            '02cute': (0, 191, 255),
            '03normal':(255, 0, 191),
            '04sleep':(191,0,255),
            '05wakeup':(191,255,0)
        }
        pose2clr = {
            '01frontal': (255,191,0),
            '02nonfrontal': (0,191,255)
        }
        ee = time.time() - ss
        print('elapsed time: ',ee )
        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
             for idx, prob in zip(final_class, final_probs)], [mc.POSE_NAMES[idx] for idx in final_pose],
            cdict=cls2clr,
        )

        _draw_point(im, final_landmarks, (0, 0, 255))

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
