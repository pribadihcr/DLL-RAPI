# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
"""Train"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os.path
import time
import numpy as np
import tensorflow as tf
import threading
import sys

from config import *
from nets import *
from six.moves import xrange
from datetime import datetime
from dataset import pascal_voc, kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeDet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def _draw_box(im, box_list, label_list, pose_list, age_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label, pose, age in zip(box_list, label_list, pose_list, age_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label+"; "+pose+"; "+age, (xmin, ymax), font, 0.3, c, 1)

def _draw_point(im, point_list, color=(0,255,0)):
    #print(point_list)
    for points in point_list:
        for p in range(0, 5):
            cv2.circle(im, (int(points[p]), int(points[p + 5])), 1, color, 2)


def _viz_prediction_result(model, images, bboxes, landmarks, labels, poses, ages, batch_det_bbox, batch_det_landmark, batch_det_pose,
                           batch_det_age, batch_det_class, batch_det_prob):
  mc = model.mc

  for i in range(len(images)):
    # draw ground truth
    _draw_box(
        images[i], bboxes[i],
        [mc.CLASS_NAMES[idx] for idx in labels[i]],
        [mc.POSE_NAMES[idx] for idx in poses[i]],
        [mc.AGE_NAMES[idx] for idx in ages[i]],
        (0, 255, 0))

    _draw_point(images[i], landmarks[i], (0, 255, 0))

    # draw prediction
    det_bbox, det_landmark, det_prob, det_pose, det_age, det_class = model.filter_prediction(
        batch_det_bbox[i], batch_det_landmark[i], batch_det_prob[i], batch_det_pose[i], batch_det_age[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.PLOT_PROB_THRESH]
    det_bbox    = [det_bbox[idx] for idx in keep_idx]
    det_landmark = [det_landmark[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]
    det_pose = [det_pose[idx] for idx in keep_idx]
    det_age = [det_age[idx] for idx in keep_idx]


    _draw_box(
        images[i], det_bbox,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob  \
            for idx, prob in zip(det_class, det_prob)], [mc.POSE_NAMES[idx] for idx in det_pose],
        [mc.AGE_NAMES[idx] for idx in det_age], (0, 0, 255))

    _draw_point(images[i], det_landmark, (0, 0, 255))

def train():
  """Train SqueezeDet model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = VGG16ConvDet(mc)
    elif FLAGS.net == 'resnet50':
      mc = kitti_res50_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = ResNet50ConvDet(mc)
    elif FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDet(mc)
    elif FLAGS.net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlus(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    def _load_data(load_to_placeholder=True):
      # read batch input
      image_per_batch, label_per_batch, pose_per_batch, age_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch, landmark_delta_per_batch, landmark_per_batch = imdb.read_batch()

      label_indices, pose_indices, age_indices, bbox_indices, box_delta_values, landmark_indices, landmark_delta_values, landmark_values, mask_indices, box_values, \
          = [], [], [], [], [], [], [], [], [], []
      aidx_set = set()
      num_discarded_labels = 0
      num_labels = 0
      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          num_labels += 1
          if (i, aidx_per_batch[i][j]) not in aidx_set:
            aidx_set.add((i, aidx_per_batch[i][j]))
            label_indices.append(
                [i, aidx_per_batch[i][j], label_per_batch[i][j]])
            pose_indices.append(
                [i, aidx_per_batch[i][j], pose_per_batch[i][j]])
            age_indices.append(
                [i, aidx_per_batch[i][j], age_per_batch[i][j]])
            mask_indices.append([i, aidx_per_batch[i][j]])
            bbox_indices.extend(
                [[i, aidx_per_batch[i][j], k] for k in range(4)])
            landmark_indices.extend([[i, aidx_per_batch[i][j], k] for k in range(mc.POINTS)])
            box_delta_values.extend(box_delta_per_batch[i][j])
            landmark_delta_values.extend(landmark_delta_per_batch[i][j])
            box_values.extend(bbox_per_batch[i][j])
            landmark_values.extend(landmark_per_batch[i][j])
          else:
            num_discarded_labels += 1

      if mc.DEBUG_MODE:
        print ('Warning: Discarded {}/({}) labels that are assigned to the same '
               'anchor'.format(num_discarded_labels, num_labels))

      if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        landmark_delta_input = model.ph_landmark_delta_input
        box_input = model.ph_box_input
        landmark_input = model.ph_landmark_input
        labels = model.ph_labels
        poses = model.ph_poses
        ages = model.ph_ages
      else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        landmark_delta_input = model.landmark_delta_input
        box_input = model.box_input
        landmark_input = model.ph_landmark_input
        labels = model.labels
        poses = model.poses
        ages = model.ages

      feed_dict = {
          image_input: image_per_batch,
          input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  [1.0]*len(mask_indices)),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          box_delta_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_delta_values),
          landmark_delta_input: sparse_to_dense(
              landmark_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.POINTS],
              landmark_delta_values),
          box_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_values),
          landmark_input: sparse_to_dense(
              landmark_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.POINTS],
              landmark_values),
          labels: sparse_to_dense(
              label_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
              [1.0]*len(label_indices)),
          poses: sparse_to_dense(
              pose_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, 2],
              [1.0] * len(pose_indices)),
          ages: sparse_to_dense(
              age_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, 2],
              [1.0] * len(age_indices)),
      }
      return feed_dict, image_per_batch, label_per_batch, pose_per_batch, age_per_batch, bbox_per_batch, landmark_per_batch

    def _enqueue(sess, coord):
      try:
        while not coord.should_stop():
          feed_dict, _, _, _, _, _, _ = _load_data()
          sess.run(model.enqueue_op, feed_dict=feed_dict)
          if mc.DEBUG_MODE:
            print ("added to the queue")
        if mc.DEBUG_MODE:
          print ("Finished enqueue")
      except Exception, e:
        coord.request_stop(e)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # list_var = [v for v in tf.global_variables() if 'conv12' in v.name or 'Unit' in v.name]
    # # list_var1 = [v for v in tf.global_variables() if 'conv12' in v.name]
    #
    # init = tf.variables_initializer(list_var)
    # sess.run(init)
    #
    # list_var = [v for v in tf.global_variables() if 'conv12' not in v.name and 'Unit' not in v.name and 'iou' not in v.name]
    # print(list_var)
    #
    # # hjkjh
    # # list_var1 = [v for v in tf.global_variables() if 'conv12' not in v.name]
    #
    # saver = tf.train.Saver(list_var)
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("training from restored model!")
    else:
        print("training a new model!")
        init = tf.global_variables_initializer()
        sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)


    coord = tf.train.Coordinator()

    if mc.NUM_THREAD > 0:
      enq_threads = []
      for _ in range(mc.NUM_THREAD):
        enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
        # enq_thread.isDaemon()
        enq_thread.start()
        enq_threads.append(enq_thread)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    run_options = tf.RunOptions(timeout_in_ms=60000)

    # try: 
    for step in xrange(FLAGS.max_steps):
      if coord.should_stop():
        sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        break

      start_time = time.time()

      if step % FLAGS.summary_step == 0:
        feed_dict, image_per_batch, label_per_batch, pose_per_batch, age_per_batch, bbox_per_batch, landmark_per_batch = \
            _load_data(load_to_placeholder=False)


        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_landmarks, model.det_probs, model.det_class,model.det_pose, model.det_age, model.conf_loss,
            model.bbox_loss, model.class_loss, model.pose_loss, model.age_loss, model.landmark_loss
        ]
        _, loss_value, summary_str, det_boxes, det_landmarks, det_probs, det_class, det_pose, det_age, \
            conf_loss, bbox_loss, class_loss, pose_loss, age_loss, landmark_loss = sess.run(
                op_list, feed_dict=feed_dict)

        #print(label_per_batch)
        #print(lbls[:,:,5])
        #print(id06)

        #dsfsd
        _viz_prediction_result(
            model, image_per_batch, bbox_per_batch, landmark_per_batch, label_per_batch, pose_per_batch, age_per_batch, det_boxes,
            det_landmarks, det_pose,  det_age, det_class, det_probs)


        image_per_batch = bgr_to_rgb(image_per_batch)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: image_per_batch})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
        summary_writer.flush()

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}, pose_loss: {}, age_loss: {}, landmark_loss: {}'.
            format(conf_loss, bbox_loss, class_loss, pose_loss, age_loss, landmark_loss))
      else:
        if mc.NUM_THREAD > 0:
          _, loss_value, conf_loss, bbox_loss, class_loss, pose_loss, age_loss, landmark_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss, model.pose_loss, model.age_loss, model.landmark_loss], options=run_options)
        else:
          feed_dict, _, _, _, _, _, _= _load_data(load_to_placeholder=False)
          _, loss_value, conf_loss, bbox_loss, class_loss, pose_loss, age_loss, landmark_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss, model.pose_loss, model.age_loss, model.landmark_loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}, pose_loss: {}, age_loss: {}, landmark_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                                                                  class_loss, pose_loss, age_loss, landmark_loss)

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
    # except Exception, e:
    #   coord.request_stop(e)
    # finally:
    #   coord.request_stop()
    #   coord.join(threads)
      tf.train.write_graph(sess.graph_def, "models/squeezenet_extraSE",
                         "models.pb", False)


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
