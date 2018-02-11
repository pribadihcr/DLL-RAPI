import sys
import os
import numpy as np
import re
import cv2
from PIL import Image
from glob import glob

this_dir = os.path.dirname(__file__)

if os.environ.get('CAFFE_MODE', False):
    sys.path.append(os.path.join(this_dir, 'BibnumberDetection_CTPN_caffe/tools/'))
    sys.path.append(os.path.join(this_dir, 'BibnumberDetection_CTPN_caffe/src/'))
    from cfg import Config as cfg
    from other import draw_boxes, resize_im, CaffeModel
    import caffe
    from detectors import TextProposalDetector, TextDetector

if os.environ.get('TENSORFLOW_MODE', False):
    import tensorflow as tf

    if os.environ.get('PERSON_MODULE', False):
        sys.path.append(os.path.join(this_dir, 'PersonDetection_FASTER_RCNN_tensorflow/'))

        from model.test import im_detect, im_detect_batch
        from model.nms_wrapper import nms
        from nets.vgg16 import vgg16
        from nets.resnet_v1 import resnetv1
        from nets.mobilenet_v1 import mobilenetv1

    if os.environ.get('FACE_MODULE', False):
        sys.path.append(os.path.join(this_dir, 'FaceDetection_MTCNN_tensorflow/'))
        import mtcnn

    if os.environ.get('BIBNUMBER_MODULE', False):
        sys.path.append(os.path.join(this_dir, 'BibnumberDetection_CTPN_tensorflow/'))
        from lib_bn_tf.networks.factory import get_network
        from lib_bn_tf.fast_rcnn.config import cfg, cfg_from_file
        from lib_bn_tf.fast_rcnn.test import test_ctpn
        from lib_bn_tf.utils.timer import Timer
        from lib_bn_tf.text_connector.detectors import TextDetector
        from lib_bn_tf.text_connector.text_connect_cfg import Config as TextLineCfg

class bibnumber_ctpn_caffe(object):
    def __init__(self, dir_model, gpu_id):
        NET_DEF_FILE = dir_model + "/bib_number/CTPN/deploy.prototxt"
        MODEL_FILE = dir_model + "/bib_number/CTPN/ctpn_trained_model.caffemodel"
        if False:  # Set this to true for CPU only mode
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(int(gpu_id))  # (cfg.TEST_GPU_ID)

        text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
        self.text_detector = TextDetector(text_proposals_detector)

        length_regexp = 'Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,'
        self.re_length = re.compile(length_regexp)

    def process(self, image, bbox):
        im_crop = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]

        im_scale, f = resize_im(im_crop, cfg.SCALE, cfg.MAX_SCALE)

        # print(np.array(im).shape)
        text_lines = self.text_detector.detect(im_scale)
        return im_scale, text_lines

class bibnumber_ctpn_tf(object):
    def __init__(self, gpu_id):
        self.session = None
        self.saver = None
        self.net = None
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    def load_model(self, dir_model):
        model_path = os.path.join(dir_model, 'bib_number/CTPN/VGGnet_fast_rcnn_iter_50000.ckpt')
        if not os.path.isfile(model_path + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(model_path + '.meta'))

        # init session
        # config = tf.ConfigProto(allow_soft_placement=True)
        # self.session = tf.Session(config=config)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            # config.gpu_options.per_process_gpu_memory_fraction = 0.8

            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

            # load network
            self.net = get_network("VGGnet_test")
            # load model
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, model_path)
            print('Loaded bib-detector model {:s}'.format(model_path))

    def process(self, image, bbox):
        def resize_im(im, scale, max_scale=None):
            f = float(scale) / min(im.shape[0], im.shape[1])
            if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
                f = float(max_scale) / max(im.shape[0], im.shape[1])
            return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

        im_crop = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        img, scale = resize_im(im_crop, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(self.session, self.net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        return img, boxes, scale

class person_faster_rcnn_tf(object):
    def __init__(self, dir_model, network, gpu_id):
        self.net = None

        if network=='res101':
            self.CLASSES = ('__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor')
        if network=='mobilenet':
            self.CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                       'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                       'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                       'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                       'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                       'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                       'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        NETS = {'mobilenet': ('mobile_faster_rcnn_iter_1190000.ckpt',), 'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # demonet = 'mobilenet'#'res101'
        tfmodel = os.path.join(dir_model, 'person_detection/faster_rcnn', NETS[network][0])
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.allow_growth = True
        #
        # # init session
        # sess = tf.Session(config=tfconfig)
        if network == 'vgg16':
            self.net = vgg16()
        elif network == 'res101':
            self.net = resnetv1(num_layers=101)
        elif network == 'mobilenet':
            self.net = mobilenetv1()
        else:
            raise NotImplementedError

        if network=='mobilenet':
            self.net.create_architecture("TEST", 81,
                                    tag='default', anchor_scales=[4, 8, 16, 32])
        if network=='res101':
            self.net.create_architecture("TEST", 21,
                                         tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(self.sess, tfmodel)
        print('Loaded network {:s}'.format(tfmodel))

        self.CONF_THRESH = 0.8
        self.NMS_THRESH = 0.3
    def process(self, image):
        scores, boxes = im_detect(self.sess, self.net, image)
        #print(boxes, scores)
        inds = []
        dets = []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            if cls == 'person':
                cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, self.NMS_THRESH)
                dets = dets[keep, :]

                inds = np.where(dets[:, -1] >= self.CONF_THRESH)[0]
                if len(inds) == 0:
                    break

        return inds, dets


class face_detection_mtcnn_tf(object):
    def __init__(self, gpu_id, session=None, gpu_fraction=None):
        self.session = session
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.20))

        self.MIN_SIZE = 20  # minimum size of face
        self.THRES = [0.6, 0.7, 0.7]  # three steps's threshold
        self.FACTOR = 0.709  # scale factor
        self.pnet_fun = None
        self.rnet_fun = None
        self.onet_fun = None
        self.total_face = 0
        self.margin = 44

    def load_model(self, model_dir):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            model_dir = os.path.join(model_dir, 'face_detection/mtcnn')
            model_path = sorted(glob(os.path.join(model_dir, '*.npy')))
            with self.session.as_default():
                MODEL_DET_P = model_path[0]
                MODEL_DET_R = model_path[1]
                MODEL_DET_O = model_path[2]
                with tf.variable_scope('pnet'):
                    data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
                    pnet = mtcnn.PNet({'data': data})
                    pnet.load(MODEL_DET_P, self.session)
                with tf.variable_scope('rnet'):
                    data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
                    rnet = mtcnn.RNet({'data': data})
                    rnet.load(MODEL_DET_R, self.session)
                with tf.variable_scope('onet'):
                    data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
                    onet = mtcnn.ONet({'data': data})
                    onet.load(MODEL_DET_O, self.session)

                self.pnet_fun = lambda img: self.session.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
                self.rnet_fun = lambda img: self.session.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
                self.onet_fun = lambda img: self.session.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                                feed_dict={'onet/input:0': img})

                print('Loaded network {:s}, {:s}, {:s}'.format(model_path[0], model_path[1], model_path[2]))

    def process(self, image):
        im = np.array(image)
        img_size = np.asarray(im.shape)[0:2]

        _ ,_ , bbs, point = mtcnn.detect_face(im[:,:,::-1], self.MIN_SIZE, self.pnet_fun, self.rnet_fun, self.onet_fun,
                                            self.THRES, self.FACTOR)

        face_boxes = []
        if len(bbs) > 0:
            for jj, bbi in enumerate(bbs):
                det = np.squeeze(bbs[jj, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                margin = self.margin
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

                #if (bb[1]) < 0 or (bb[3]) < 0 or (bb[0]) < 0 or (bb[2]) < 0:
                #    continue
                # face = im[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
                face_boxes.append(bb)
        return face_boxes

    def process_find_largest_face(self, filepath):
        head, tail = os.path.split(filepath)
        basename, file_extension = os.path.splitext(tail)

        im = Image.open(filepath)
        (width, height) = im.size
        im = np.array(im)

        # cv_img = im.astype(np.uint8)

        img_size = np.asarray(im.shape)[0:2]

        try:
            _, _, bbs, points = mtcnn.detect_face(im, self.MIN_SIZE, self.pnet_fun, self.rnet_fun, self.onet_fun,
                                            self.THRES, self.FACTOR)
        except:
            print("detect error!")
            sys.exit(1)

        bbc = np.zeros(4, dtype=np.int32)
        choosedface = False
        if len(bbs) > 0:

            maxarea = 0
            for jj, bbi in enumerate(bbs):
                det = np.squeeze(bbs[jj, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bd = np.zeros(4, dtype=np.int32)
                W = det[2] - det[0]
                H = det[3] - det[1]
                margin1 = W * 0.2
                margin2 = H * 0.2
                bb[0] = np.maximum(det[0] - margin1 / 2, 0)
                bb[1] = np.maximum(det[1] - margin2 / 2, 0)
                bb[2] = np.minimum(det[2] + margin1 / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin2 / 2, img_size[0])

                W = bb[2] - bb[0]
                H = bb[3] - bb[1]

                # if (bb[1]) < 0 or (bb[3]) > height or (bb[0]) < 0 or (bb[2]) > width:
                #     continue
                if W * H > maxarea:
                    maxarea = W * H
                    bbc[0] = bb[0]
                    bbc[1] = bb[1]
                    bbc[2] = bb[2]
                    bbc[3] = bb[3]
                    choosedface = True
                    # bbd[0] = bd[0]
                    # bbd[1] = bd[1]
                    # bbd[2] = bd[2]
                    # bbd[3] = bd[3]
            # cv2.rectangle(im, (bbc[0], bbc[1]), (bbc[2], bbc[3]), (0, 0, 255), 2)

            # cv2.rectangle(im, (160,120), (480,360), (255, 0, 0), 2)
            if choosedface:
                cropped = im[bbc[1]:bbc[3], bbc[0]:bbc[2], :]
                # if not os.path.isdir(os.path.join(src_path, folder)):
                #    os.makedirs(os.path.join(src_path, folder))
                # print(basename)
                # cv2.imwrite(basename + '_face.png', cropped[:, :, ::-1])
                return cropped[:,:,::-1]
            else:
                print("Face is not detected or discarded!!!")
                return None
        return None
