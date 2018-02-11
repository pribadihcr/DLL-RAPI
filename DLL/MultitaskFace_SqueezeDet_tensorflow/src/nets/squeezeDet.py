# Author: Bichen Wu (bichen@berkeley.edu) 085016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton
#import tensorflow.contrib.slim as slim
from tensorpack import logger, QueueInput, InputDesc, PlaceholderInput, TowerContext
# from tensorpack.models import *
# from tensorpack.callbacks import *
# from tensorpack.train import *
# from tensorpack.dataflow import imgaug
# from tensorpack.tfutils import argscope, get_model_loader, varreplace
# from tensorpack.tfutils.scope_utils import under_name_scope


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
LinearWrap)

# sbnet_module = tf.load_op_library('libsbnet.so')

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()



  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    def gaussian_noise_layer(input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    # @layer_register(log_shape=True)
    # def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
    #               W_init=None, nl=tf.identity):
    #     in_shape = x.get_shape().as_list()
    #     in_channel = in_shape[3]  # in_shape[1]
    #     assert out_channel % in_channel == 0
    #     channel_mult = out_channel // in_channel
    #
    #     if W_init is None:
    #         W_init = tf.contrib.layers.variance_scaling_initializer()
    #     kernel_shape = [kernel_shape, kernel_shape]
    #     filter_shape = kernel_shape + [in_channel, channel_mult]
    #
    #     W = tf.get_variable('W', filter_shape, initializer=W_init)
    #     conv = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding=padding, data_format='NHWC')
    #     return nl(conv, name='output')
    #
    # @under_name_scope()
    # def channel_shuffle(l, group):
    #     # l = tf.transpose(l, [0,3,1,2])
    #     in_shape = l.get_shape().as_list()
    #     in_channel = in_shape[3]  # in_shape[1]
    #     l = tf.reshape(l, [-1, in_shape[1], in_shape[2], group, in_channel // group])
    #     l = tf.transpose(l, [0, 1, 2, 4, 3])
    #     l = tf.reshape(l, [-1, in_shape[1], in_shape[2], in_channel])
    #     # l = tf.transpose(l, [0, 2, 3, 1]) #NHWC
    #
    #     return l
    #
    # def BN(x, name):
    #     return BatchNorm('bn', x)
    #
    # def shufflenet_unit(l, out_channel, group, stride):
    #     in_shape = l.get_shape().as_list()
    #     in_channel = in_shape[3]  # in_shape[1]
    #     shortcut = l
    #
    #     # We do not apply group convolution on the first pointwise layer
    #     # because the number of input channels is relatively small.
    #     first_split = group if in_channel != 16 else 1
    #     l = Conv2D('conv1', l, out_channel // 4, 1, split=first_split, nl=BNReLU)
    #     l = channel_shuffle(l, group)
    #     l = DepthConv('dconv', l, out_channel // 4, 3, nl=BN, stride=stride)
    #     l = Conv2D('conv2', l,
    #                out_channel if stride == 1 else out_channel - in_channel,
    #                1, split=group, nl=BN)
    #     if stride == 1:  # unit (b)
    #         output = tf.nn.relu(shortcut + l)
    #     else:  # unit (c)
    #         shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
    #         output = tf.concat([shortcut, tf.nn.relu(l)], axis=3)  # axis=1)
    #     return output
    # def group_conv(net, output, stride, group, relu=True, scope="GConv"):
    #   assert 0 == output % group, "Output channels must be a multiple of groups"
    #   num_channels_in_group = output // group
    #   with tf.variable_scope(scope):
    #       net = tf.split(net, group, axis=3, name="split")
    #       for i in range(group):
    #           net[i] = slim.conv2d(net[i],
    #                                num_channels_in_group,
    #                                [1, 1],
    #                                stride=stride,
    #                                activation_fn=tf.nn.relu if relu else None,
    #                                normalizer_fn=slim.batch_norm,
    #                                normalizer_params={'is_training': True})
    #       net = tf.concat(net, axis=3, name="concat")
    #
    #   return net
    #
    # def channel_shuffle(net, output, group, scope="ChannelShuffle"):
    #   assert 0 == output % group, "Output channels must be a multiple of groups"
    #   num_channels_in_group = output // group
    #   with tf.variable_scope(scope):
    #       net = tf.split(net, output, axis=3, name="split")
    #       chs = []
    #       for i in range(group):
    #           for j in range(num_channels_in_group):
    #               chs.append(net[i + j * group])
    #       net = tf.concat(chs, axis=3, name="concat")
    #
    #   return net
    #
    # def shuffle_bottleneck(net, output, stride, group=1, scope="Unit"):
    #   if 1 != stride:
    #       _b, _h, _w, _c = net.get_shape().as_list()
    #       output = output - _c
    #
    #   assert 0 == output % group, "Output channels must be a multiple of groups"
    #
    #   with tf.variable_scope(scope):
    #       if 1 != stride:
    #           net_skip = slim.avg_pool2d(net, [3, 3], stride, padding="SAME", scope='3x3AVGPool')
    #       else:
    #           net_skip = net
    #
    #       net = group_conv(net, output, 1, group, relu=True, scope="1x1ConvIn")
    #
    #
    #       net = channel_shuffle(net, output, group, scope="ChannelShuffle")
    #
    #       with tf.variable_scope("3x3DWConv"):
    #           depthwise_filter = tf.get_variable("depth_conv_w",
    #                                              [3, 3, output, 1],
    #                                              initializer=tf.truncated_normal_initializer())
    #           net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")
    #
    #       net = group_conv(net, output, 1, group, relu=True, scope="1x1ConvOut")
    #
    #       if 1 != stride:
    #           net = tf.concat([net, net_skip], axis=3)
    #       else:
    #           net = net + net_skip
    #
    #   return net
    #
    # def _depthwise_separable_conv(inputs,
    #                               num_pwc_filters,
    #                               width_multiplier,
    #                               sc,
    #                               downsample=False):
    #     """ Helper function to build the depth-wise separable convolution layer.
    #     """
    #     num_pwc_filters = round(num_pwc_filters * width_multiplier)
    #     _stride = 2 if downsample else 1
    #
    #     # skip pointwise by setting num_outputs=None
    #     depthwise_conv = slim.separable_convolution2d(inputs,
    #                                                   num_outputs=None,
    #                                                   stride=_stride,
    #                                                   depth_multiplier=1,
    #                                                   kernel_size=[3, 3],
    #                                                   scope=sc + '/depthwise_conv')
    #
    #     bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
    #     pointwise_conv = slim.convolution2d(bn,
    #                                         num_pwc_filters,
    #                                         kernel_size=[1, 1],
    #                                         scope=sc + '/pointwise_conv')
    #     bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
    #     return bn

    # noise = gaussian_noise_layer(self.image_input, .5)
    def resnet_shortcut(l, n_out, stride, nl=tf.identity):
        #data_format = get_arg_scope()['Conv2D']['data_format']
        n_in = l.get_shape().as_list()[3]
        if n_in != n_out:  # change dimension when channel is not the same
            return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
        else:
            return l

    def get_bn(zero_init=False):
        """
        Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
        """
        if zero_init:
            return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
        else:
            return lambda x, name: BatchNorm('bn', x)

    def se_resnet_bottleneck(l, ch_out, stride):
        shortcut = l
        l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
        print(l.get_shape())
        l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
        print(l.get_shape())
        l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
        print(l.get_shape())


        squeeze = GlobalAvgPooling('gap', l)
        squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
        squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
        #data_format = get_arg_scope()['Conv2D']['data_format']
        ch_ax = 3
        shape = [-1, 1, 1, 1]
        shape[ch_ax] = ch_out * 4
        l = l * tf.reshape(squeeze, shape)

        return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    net = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

    # Two extra fire modules that are not trained before
    # fire10 = self._fire_layer(
    #     'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    # fire11 = self._fire_layer(
    #     'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    # dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')
    #
    # print(dropout11.get_shape())

    # with tf.variable_scope('ExtraNet') as sc:
    #     end_points_collection = sc.name + '_end_points'
    #     with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
    #                         activation_fn=None,
    #                         outputs_collections=[end_points_collection]):
    #         with slim.arg_scope([slim.batch_norm],
    #                             is_training=True,
    #                             activation_fn=tf.nn.relu):
    #             width_multiplier = 1
    #             fire10 = self._fire_layer('fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    #             net = _depthwise_separable_conv(fire10, 768, width_multiplier, sc='conv_ds_1')
    #             fire11 = self._fire_layer('fire11', net, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    #             net = _depthwise_separable_conv(fire11, 768, width_multiplier, sc='conv_ds_2')
    #             print(net.get_shape())

    # fire10 = self._fire_layer('fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    # net = group_conv(fire10, 768, 1, 3, relu=True, scope="conv_g_1")
    # # fire11 = self._fire_layer('fire11', net, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    # net = group_conv(net, 768, 1, 3, relu=True, scope="conv_g_2")
    # print(net.get_shape())

    # net = self._fire_layer('fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    with TowerContext('', is_training=True):
        for i in range(2):
            with tf.variable_scope('block{}'.format(i)):
                net = se_resnet_bottleneck(net, 192, 1)
        print(net.get_shape())

    # with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'), \
    #      argscope(Conv2D, use_bias=False):
    #     group = 3
    #     channels = [768]
    #
    #     with TowerContext('', is_training=True):
    #         # with varreplace.freeze_variables():
    #         with tf.variable_scope('group1'):
    #             for i in range(2):
    #                 with tf.variable_scope('block{}'.format(i)):
    #                     net = shufflenet_unit(net, channels[0], group, 1)

    # net = shuffle_bottleneck(fire11, 768, 1, 3, scope='Unit-1')
    # net = shuffle_bottleneck(net, 768, 1, 3, scope='Unit-2')
    # print(net.get_shape())

    net = tf.nn.dropout(net, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4 + mc.POINTS + 2 + 2)
    self.preds = self._conv_layer(
        'conv12', net, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

    # variables_to_restore = slim.get_variables_to_restore()
    # print(variables_to_restore)

    modelparams1 = self.model_params
    for varres in tf.trainable_variables():#variables_to_restore:
        if varres not in modelparams1:
            if 'iou' not in varres.name:
                self.model_params += [varres]

    print(self.model_params)

    total_parameters = 0
    #iterating over all variables
    for variable in tf.trainable_variables():
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        print(variable.name, local_parameters)
        total_parameters+=local_parameters
    print(total_parameters)

    # list_var = [v.name for v in tf.global_variables() if 'conv12' in v.name or 'Unit' in v.name]
    # print(list_var)
    # sdf

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
