import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import tf_util
from PWCLO_util import *


def placeholder_inputs(batch_size, num_point):

    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 3))

    q_gt = tf.placeholder(tf.float32, shape=(batch_size, 4))
    t_gt = tf.placeholder(tf.float32, shape=(batch_size, 3,1))
    
    return pointclouds_pl, q_gt, t_gt



def get_model(point_cloud, is_training, bn_decay=None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_points_f1 = point_cloud[:, :num_point, 3:]

    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_points_f2 = point_cloud[:, num_point:, 3:]


    with tf.variable_scope('sa1') as scope:

        l0_xyz_f1, l0_points_f1 = pointnet_sa_module(l0_xyz_f1 , l0_points_f1, npoint=2048, nsample=32, mlp=[8,8,16], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer0')

        l1_xyz_f1, l1_points_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=1024, nsample=32, mlp=[16,16,32], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz_f1, l2_points_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=256, nsample=16, mlp=[32,32,64], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        
        l3_xyz_f1, l3_points_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1, npoint=64, nsample=16, mlp=[64,64,128], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3')


        scope.reuse_variables()
        
        l0_xyz_f2, l0_points_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=2048, nsample=32, mlp=[8,8,16],   mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer0')

        l1_xyz_f2, l1_points_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=1024, nsample=32, mlp=[16,16,32],   mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz_f2, l2_points_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=256, nsample=16, mlp=[32,32,64], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz_f2, l3_points_f2 = pointnet_sa_module(l2_xyz_f2, l2_points_f2, npoint=64, nsample=16, mlp=[64,64,128], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3')


    l2_points_f1_new = cost_volume(l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2,  nsample=4, nsample_q=32, mlp1=[128,64,64], mlp2 = [128,64], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_2', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1_cost_volume = pointnet_sa_module(l2_xyz_f1, l2_points_f1_new, npoint=64, nsample=16, mlp=[128, 64, 64], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3_flow')


    #####layer3#############################################
    
    # Feature Propagation
    
    l3_points_predict = l3_points_f1_cost_volume

    l3_cost_volume_w = flow_predictor( l3_points_f1, None, l3_points_predict, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l3_costvolume_predict_ww')
    
    W_l3_feat1 =  tf.nn.softmax(l3_cost_volume_w, dim=1)

    l3_points_f1_new = tf.reduce_sum(l3_points_predict*W_l3_feat1, axis = 1, keep_dims = True)
    l3_points_f1_new_big = tf_util.conv1d(l3_points_f1_new, 256, 1, padding='VALID', activation_fn=None, scope='l3_big')
    
    l3_points_f1_new_q = tf.layers.dropout(l3_points_f1_new_big, rate = 0.5, training = is_training)
    l3_points_f1_new_t = tf.layers.dropout(l3_points_f1_new_big, rate = 0.5, training = is_training)

    l3_q_coarse = tf_util.conv1d(l3_points_f1_new_q, 4, 1, padding='VALID', activation_fn=None, scope='l3_q_coarse')
    l3_q_coarse = l3_q_coarse / (tf.sqrt(tf.reduce_sum(l3_q_coarse*l3_q_coarse, axis=-1, keep_dims=True)+1e-10) + 1e-10)

    l3_t_coarse = tf_util.conv1d(l3_points_f1_new_t, 3, 1, padding='VALID', activation_fn=None, scope='l3_t_coarse')


    l3_q = tf.squeeze(l3_q_coarse)
    l3_t = tf.squeeze(l3_t_coarse)


    #####layer 2##############################################################

    l2_q_coarse = tf.reshape(l3_q, [batch_size, 1, -1])
    l2_t_coarse = tf.reshape(l3_t, [batch_size, 1, -1])

    l2_q_inv = inv_q(l2_q_coarse)

    # # warped flow
    pc1_sample_256_q = tf.concat([tf.zeros([batch_size, 256, 1]), l2_xyz_f1], axis = -1)
    l2_flow_warped = mul_q_point(l2_q_coarse, pc1_sample_256_q, batch_size)
    l2_flow_warped = tf.slice(mul_point_q(l2_flow_warped, l2_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l2_t_coarse
    
    # get the cost volume 
    l2_cost_volume = cost_volume(l2_flow_warped, l2_points_f1, l2_xyz_f2, l2_points_f2,  nsample=4, nsample_q=6, mlp1=[128,64,64], mlp2=[128,64], is_training=is_training, bn_decay=bn_decay, scope='l2_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1
   
    l2_cost_volume_w_upsample = set_upconv_module(l2_xyz_f1, l3_xyz_f1, l2_points_f1, l3_cost_volume_w, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l2w', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_cost_volume_upsample = set_upconv_module(l2_xyz_f1, l3_xyz_f1, l2_points_f1, l3_points_predict, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l2costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)


    l2_cost_volume_predict = flow_predictor( l2_points_f1, l2_cost_volume_upsample, l2_cost_volume, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l2_costvolume_predict')
    l2_cost_volume_w = flow_predictor( l2_cost_volume_w_upsample, l2_points_f1, l2_cost_volume_predict, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l2_w_predict')

    W_l2_cost_volume =  tf.nn.softmax(l2_cost_volume_w, dim=1)


    l2_cost_volume_sum = tf.reduce_sum(l2_cost_volume_predict * W_l2_cost_volume, axis = 1, keep_dims = True)
    l2_cost_volume_sum_big = tf_util.conv1d(l2_cost_volume_sum, 256, 1, padding='VALID', activation_fn=None, scope='l2_big')
    
    l2_cost_volume_sum_q = tf.layers.dropout(l2_cost_volume_sum_big, rate = 0.5, training = is_training)
    l2_cost_volume_sum_t = tf.layers.dropout(l2_cost_volume_sum_big, rate = 0.5, training = is_training)


    l2_q_det = tf_util.conv1d(l2_cost_volume_sum_q, 4, 1, padding='VALID', activation_fn=None, scope='l2_q_det')
    l2_q_det = l2_q_det / (tf.sqrt(tf.reduce_sum(l2_q_det*l2_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l2_t_det = tf_util.conv1d(l2_cost_volume_sum_t, 3, 1, padding='VALID', activation_fn=None, scope='l2_t_det')

    l2_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l2_t_coarse], axis = -1)
    l2_t_coarse_trans = mul_q_point(l2_q_coarse, l2_t_coarse_trans, batch_size)
    l2_t_coarse_trans = tf.slice(mul_point_q(l2_t_coarse_trans, l2_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_1
    
    l2_q = tf.squeeze(mul_point_q(l2_q_det, l2_q_coarse, batch_size))
    l2_t = tf.squeeze(l2_t_coarse_trans + l2_t_det)

    
    ########layer 1#####################################


    l1_q_coarse = tf.reshape(l2_q, [batch_size, 1, -1])
    l1_t_coarse = tf.reshape(l2_t, [batch_size, 1, -1])

    l1_q_inv = inv_q(l1_q_coarse)

    #  warped flow
    pc1_sample_1024_q = tf.concat([tf.zeros([batch_size, 1024, 1]), l1_xyz_f1], axis = -1)

    l1_flow_warped = mul_q_point(l1_q_coarse, pc1_sample_1024_q, batch_size)
    l1_flow_warped = tf.slice(mul_point_q(l1_flow_warped, l1_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l1_t_coarse
    
    # get the cost volume 
    l1_cost_volume = cost_volume(l1_flow_warped, l1_points_f1, l1_xyz_f2, l1_points_f2,  nsample=4, nsample_q=6, mlp1=[128,64,64], mlp2=[128,64], is_training=is_training, bn_decay=bn_decay, scope='l1_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1

    l1_cost_volume_w = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_cost_volume_w, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l1w', is_training=is_training, bn_decay=bn_decay, knn=True)
    l1_cost_volume_up_sample= set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_cost_volume_predict, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l1costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)

    l1_cost_volume_predict = flow_predictor( l1_points_f1, l1_cost_volume_up_sample, l1_cost_volume, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l1_costvolume_predict')

    l1_cost_volume_w = flow_predictor( l1_cost_volume_w, l1_points_f1, l1_cost_volume_predict, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l1_w_predict')

    W_l1_cost_volume =  tf.nn.softmax(l1_cost_volume_w, dim=1)


    l1_cost_volume_8 = tf.reduce_sum(l1_cost_volume_predict*W_l1_cost_volume, axis = 1, keep_dims = True)
    
    l1_cost_volume_sum_big = tf_util.conv1d(l1_cost_volume_8, 256, 1, padding='VALID', activation_fn=None, scope='l1_big')
    
    l1_cost_volume_sum_q = tf.layers.dropout(l1_cost_volume_sum_big, rate = 0.5, training = is_training)
    l1_cost_volume_sum_t = tf.layers.dropout(l1_cost_volume_sum_big, rate = 0.5, training = is_training)


    l1_q_det = tf_util.conv1d(l1_cost_volume_sum_q, 4, 1, padding='VALID', activation_fn=None, scope='l1_q_det')
    l1_q_det = l1_q_det / (tf.sqrt(tf.reduce_sum(l1_q_det*l1_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l1_t_det = tf_util.conv1d(l1_cost_volume_sum_t, 3, 1, padding='VALID', activation_fn=None, scope='l1_t_det')

    l1_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l1_t_coarse], axis = -1)
    l1_t_coarse_trans = mul_q_point(l1_q_coarse, l1_t_coarse_trans, batch_size)
    l1_t_coarse_trans = tf.slice(mul_point_q(l1_t_coarse_trans, l1_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_1

    l1_q = tf.squeeze(mul_point_q(l1_q_det, l1_q_coarse, batch_size))
    l1_t = tf.squeeze(l1_t_coarse_trans + l1_t_det)


    ########layer 0#####################################


    l0_q_coarse = tf.reshape(l1_q, [batch_size, 1, -1])
    l0_t_coarse = tf.reshape(l1_t, [batch_size, 1, -1])

    l0_q_inv = inv_q(l0_q_coarse)

    # # warped  flow
    pc1_sample_2048_q = tf.concat([tf.zeros([batch_size, 2048, 1]), l0_xyz_f1], axis = -1)

    l0_flow_warped = mul_q_point(l0_q_coarse, pc1_sample_2048_q, batch_size)
    l0_flow_warped = tf.slice(mul_point_q(l0_flow_warped, l0_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l0_t_coarse
    
    # get the cost volume 
    l0_cost_volume = cost_volume(l0_flow_warped, l0_points_f1, l0_xyz_f2, l0_points_f2,  nsample=4, nsample_q=6, mlp1=[128,64,64], mlp2=[128,64], is_training=is_training, bn_decay=bn_decay, scope='l0_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1

    l0_cost_volume_w = set_upconv_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_cost_volume_w, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l0w', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_cost_volume_up_sample= set_upconv_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_cost_volume_predict, nsample=8, mlp=[128,64], mlp2=[64], scope='up_sa_layer_layer_l0costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)

    l0_cost_volume_predict = flow_predictor( l0_points_f1, l0_cost_volume_up_sample, l0_cost_volume, mlp=[128,64], is_training = is_training , bn_decay = bn_decay, scope='l0_costvolume_predict')

    W_l0_cost_volume =  tf.nn.softmax(l0_cost_volume_w, dim=1)


    l0_cost_volume_8 = tf.reduce_sum(l0_cost_volume_predict*W_l0_cost_volume, axis = 1, keep_dims = True)
    
    l0_cost_volume_sum_big = tf_util.conv1d(l0_cost_volume_8, 256, 1, padding='VALID', activation_fn=None, scope='l0_big')
    
    l0_cost_volume_sum_q = tf.layers.dropout(l0_cost_volume_sum_big, rate = 0.5, training = is_training)
    l0_cost_volume_sum_t = tf.layers.dropout(l0_cost_volume_sum_big, rate = 0.5, training = is_training)


    l0_q_det = tf_util.conv1d(l0_cost_volume_sum_q, 4, 1, padding='VALID', activation_fn=None, scope='l0_q_det')
    l0_q_det = l0_q_det / (tf.sqrt(tf.reduce_sum(l0_q_det*l0_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l0_t_det = tf_util.conv1d(l0_cost_volume_sum_t, 3, 1, padding='VALID', activation_fn=None, scope='l0_t_det')
    l0_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l0_t_coarse], axis = -1)
    l0_t_coarse_trans = mul_q_point(l0_q_coarse, l0_t_coarse_trans, batch_size)
    l0_t_coarse_trans = tf.slice(mul_point_q(l0_t_coarse_trans, l0_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_1

    l0_q = tf.squeeze(mul_point_q(l0_q_det, l0_q_coarse, batch_size))
    l0_t = tf.squeeze(l0_t_coarse_trans + l0_t_det)

    l0_q_norm = l0_q / (tf.sqrt(tf.reduce_sum(l0_q*l0_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l1_q_norm = l1_q / (tf.sqrt(tf.reduce_sum(l1_q*l1_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l2_q_norm = l2_q / (tf.sqrt(tf.reduce_sum(l2_q*l2_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l3_q_norm = l3_q / (tf.sqrt(tf.reduce_sum(l3_q*l3_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)


    return  l0_q_norm, l0_t, l1_q_norm, l1_t, l2_q_norm, l2_t, l3_q_norm, l3_t



def get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, qq_gt, t_gt, w_x, w_q):#####idx来选择真值

    t_gt = tf.squeeze(t_gt)###  8,3

    l0_q_norm = l0_q / (tf.sqrt(tf.reduce_sum(l0_q*l0_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l0_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((qq_gt-l0_q_norm)*(qq_gt-l0_q_norm), axis=-1, keep_dims=True)+1e-10)) 
    l0_loss_x = tf.reduce_mean( tf.sqrt((l0_t-t_gt) * (l0_t-t_gt)+1e-10))
    
    l0_loss = l0_loss_x * tf.exp(-w_x) + w_x + l0_loss_q * tf.exp(-w_q) + w_q
    
    tf.summary.scalar('l0 loss', l0_loss)


    l1_q_norm = l1_q / (tf.sqrt(tf.reduce_sum(l1_q*l1_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l1_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((qq_gt-l1_q_norm)*(qq_gt-l1_q_norm), axis=-1, keep_dims=True)+1e-10)) 
    l1_loss_x = tf.reduce_mean(tf.sqrt((l1_t-t_gt) * (l1_t-t_gt)+1e-10))
    
    l1_loss = l1_loss_x * tf.exp(-w_x) + w_x + l1_loss_q * tf.exp(-w_q) + w_q
    
    tf.summary.scalar('l1 loss', l1_loss)


    l2_q_norm = l2_q / (tf.sqrt(tf.reduce_sum(l2_q*l2_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l2_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((qq_gt-l2_q_norm)*(qq_gt-l2_q_norm), axis=-1, keep_dims=True)+1e-10))
    l2_loss_x = tf.reduce_mean(tf.sqrt((l2_t-t_gt) * (l2_t-t_gt)+1e-10))
    
    l2_loss = l2_loss_x * tf.exp(-w_x) + w_x + l2_loss_q * tf.exp(-w_q) + w_q

    tf.summary.scalar('l2 loss', l2_loss)

    l3_q_norm = l3_q / (tf.sqrt(tf.reduce_sum(l3_q*l3_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l3_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((qq_gt-l3_q_norm)*(qq_gt-l3_q_norm), axis=-1, keep_dims=True)+1e-10))
    l3_loss_x = tf.reduce_mean(tf.sqrt((l3_t-t_gt) * (l3_t-t_gt)+1e-10))
    
    l3_loss = l3_loss_x * tf.exp(-w_x) + w_x + l3_loss_q * tf.exp(-w_q) + w_q

    tf.summary.scalar('l3 loss', l3_loss)

    loss_sum = 1.6*l3_loss + 0.8*l2_loss + 0.4*l1_loss + 0.2*l0_loss

    tf.add_to_collection('losses', loss_sum)

    return loss_sum


