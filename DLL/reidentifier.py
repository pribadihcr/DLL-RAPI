import sys
import os
import cv2
import imageio
import csv
import ntpath
import matplotlib.pyplot as plt
from time import time as timer

this_dir = os.path.dirname(__file__)

if os.environ.get('TENSORFLOW_MODE', False):
    import tensorflow as tf
    # sys.path.append(os.path.join(this_dir, 'TrackingByReID_DEEPSORT_tensorflow/'))
    from TrackingByReID_DEEPSORT_tensorflow import nn_matching
    from TrackingByReID_DEEPSORT_tensorflow.tracker import Tracker
    from TrackingByReID_DEEPSORT_tensorflow.reid import encoder as enc

class trackbyreid_deepsort_tf(object):
    def __init__(self, gpu_id, session=None, gpu_fraction=None):
        self.session = session
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.40))
        self.tracker = None
        self.encoder = None

    def load_model(self, model_dir):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            model_path = os.path.join(model_dir, 'person_re_id/deep_sort/mars-small128.ckpt-68577')
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", 0.2, 100)
            self.tracker = Tracker(metric)
            self.encoder = enc.create_box_encoder(model_path)
            print('Loaded network {:s}'.format(model_path))

    def process(self, FLAGS, video, output_dir, darknet):
        if FLAGS.BK_MOG and FLAGS.track:
            fgbg = cv2.createBackgroundSubtractorMOG2()
        vid = imageio.get_reader(video, 'ffmpeg')
        if FLAGS.csv:
            f = open(os.path.join(output_dir, '{}.csv'.format(ntpath.basename(video)[:-4])), 'w')
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['frame_id', 'track_id', 'x', 'y', 'w', 'h'])
            f.flush()
        else:
            f = None
            writer = None

        # buffers for demo in batch
        buffer_inp = list()
        buffer_pre = list()

        elapsed = 0
        start = timer()
        # self.say('Press [ESC] to quit demo')
        # postprocessed = []
        # Loop through frames
        n = 0

        frame = vid.get_data(0)
        if FLAGS.display:
            plt.ion()
            fig = plt.figure()
            ax = plt.gca()
            img_artist = ax.imshow(frame)

        for num in range(1, 20000):
            try:
                frame = vid.get_data(num)
                print(num)
            except:
                break
            elapsed += 1
            # _, frame = camera.read()
            if frame is None:
                print ('\nEnd of Video')
                break
            if FLAGS.skip != n:
                n += 1
                continue
            n = 0
            if FLAGS.BK_MOG and FLAGS.track:
                fgmask = fgbg.apply(frame)
            else:
                fgmask = None

            preprocessed = darknet.framework.preprocess(frame)

            buffer_inp.append(frame)
            buffer_pre.append(preprocessed)
            # Only process and imshow when queue is full
            if elapsed % FLAGS.queue == 0:
                feed_dict = {darknet.inp: buffer_pre}
                net_out = darknet.sess.run(darknet.out, feed_dict)

                for img, single_out in zip(buffer_inp, net_out):
                    if not FLAGS.track:
                        postprocessed, _, _ = darknet.framework.postprocess(
                            single_out, img)

                    else:

                        postprocessed, bbox, id_num = darknet.framework.postprocess(
                            single_out, img, frame_id=elapsed,
                            csv_file=f, csv=writer, mask=fgmask,
                            encoder=self.encoder, tracker=self.tracker)

                        if id_num != -1:
                            if FLAGS.saveBox:
                                if not os.path.exists(os.path.join(output_dir, 'ids', str(id_num))):
                                    os.makedirs(os.path.join(output_dir, 'ids', str(id_num)))
                                cv2.imwrite(os.path.join(output_dir, 'ids', str(id_num), 'frame_%d' % num + '.jpg'),
                                            img[int(bbox[1]):(int(bbox[3])), int(bbox[0]):(int(bbox[2])), :][:, :, ::-1])

                    if FLAGS.display:
                        # cv2.imshow('', postprocessed)
                        img_artist.set_data(postprocessed)
                        plt.show()
                        plt.pause(0.00001)
                # Clear Buffers
                buffer_inp = list()
                buffer_pre = list()

            if elapsed % 5 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('{0:3.3f} FPS'.format(
                    elapsed / (timer() - start)))
                sys.stdout.flush()

        sys.stdout.write('\n')

        if FLAGS.csv:
            f.close()