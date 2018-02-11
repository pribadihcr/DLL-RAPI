import sys
import os

this_dir = os.path.dirname(__file__)
# print(this_dir)
# # Add lib to PYTHONPATH
# lib_path = os.path.join(this_dir, 'BibnumberRecogition_CTPN_caffe/tools/')
# add_path(lib_path)
# lib_path = os.path.join(this_dir, 'BibnumberRecogition_CTPN_caffe/src/')
# add_path(lib_path)

if os.environ.get('PYTORCH_MODE', False):
    sys.path.append(os.path.join(this_dir, 'BibnumberRecognition_CRNN_pytorch/'))
    import utils as utils
    import dataset as dataset
    from PIL import Image, ImageDraw, ImageFont
    import models.crnn as crnn
    import torch
    # import json
    # import torch.backends.cudnn as cudnn
    # import torch.optim as optim
    import torch.utils.data
    from torch.autograd import Variable

if os.environ.get('TENSORFLOW_MODE', False):
    import tensorflow as tf

class age_gender_tf(object):
    def __init__(self, dir_model, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.gpuid = gpu_id
        tfmodel = os.path.join(dir_model, 'age_gender/rothe', 'model.ckpt-14001')
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            # config.gpu_options.per_process_gpu_memory_fraction = 0.8
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
            self.images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
            images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl)  # BGR TO RGB
            images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
            self.train_mode = tf.placeholder(tf.bool)
            age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                         phase_train=train_mode,
                                                                         weight_decay=1e-5)
            self.gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
            age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            self.age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self.sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(self.sess, tfmodel)
            print('Loaded network {:s}'.format(tfmodel))

    def process(self, face_image):
        age, gender = self.sess.run([self.age, self.gender], feed_dict={self.images_pl: face_image, self.train_mode: False})
        return age, gender

class bibnumber_crnn_pytorch(object):
    def __init__(self, dir_model, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.gpuid = gpu_id
        self.model, self.converter = self.crnnSource(dir_model)

    def crnnSource(self, dir_model):
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'  # keys.alphabet
        converter = utils.strLabelConverter(alphabet)

        if self.gpuid == '-1':
            print('cpu-version')
            model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
        else:
            print('gpu-version')
            model = crnn.CRNN(32, 1, len(alphabet) + 1, 256).cuda()
        path = dir_model + '/bib_number/CRNN/crnn.pth'  # './crnn/samples/netCRNN63.pth'
        model.load_state_dict(torch.load(path))
        print('Loaded network {:s}'.format(path))
        return model, converter

    def process(self, im, text_recs):
        index = 0
        sim_preds = []
        for rec in text_recs:

            if len(rec) > 8:
                top, left, bottom, right, score = rec[0], rec[1], rec[6], rec[7], rec[8]
            else:
                top, left, bottom, right, score = rec
            crop_img = im[int(left):int(right), int(top):int(bottom)]
            # pt1 = (rec[0], rec[1])
            # pt2 = (rec[2], rec[3])
            # pt3 = (rec[6], rec[7])
            # pt4 = (rec[4], rec[5])
            # partImg = dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
            # # mahotas.imsave('%s.jpg'%index, partImg)


            image = Image.fromarray(crop_img).convert('L')
            # height,width,channel=partImg.shape[:3]
            # print(height,width,channel)
            # print(image.size)

            # image = Image.open('./img/t4.jpg').convert('L')
            scale = image.size[1] * 1.0 / 32
            w = image.size[0] / scale
            w = int(w)
            # print(w)

            transformer = dataset.resizeNormalize((w, 32))
            if self.gpuid == '-1':
                image = transformer(image)
            else:
                image = transformer(image).cuda()
            image = image.view(1, *image.size())
            image = Variable(image)
            self.model.eval()
            preds = self.model(image)
            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
            # print('%-20s => %-20s' % (raw_pred, sim_pred))
            # print(index)
            # print(sim_pred)
            sim_preds.append(sim_pred)
            # index = index + 1
        return sim_preds