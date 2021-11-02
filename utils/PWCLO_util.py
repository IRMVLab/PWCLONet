import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/new_grouping'))#####
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/ops_square'))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point

import tensorflow as tf
import numpy as np
import tf_util


def inv_q(q):
    
    q = tf.squeeze(q, axis = 1)

    q_2 = tf.reduce_sum(q*q, axis = -1, keep_dims = True) + 1e-10
    q_  = tf.concat([tf.slice(q, [0, 0], [-1, 1]), -tf.slice(q, [0, 1], [-1, 3])], axis = -1)
    q_inv = q_/q_2

    return q_inv


def mul_point_q(q_a, q_b, batch_size):

    q_b = tf.reshape(q_b, [batch_size, 1, 4])
    
    q_result_0 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 0])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 1])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 3])
    q_result_0 = tf.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 3])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 2])
    q_result_1 = tf.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 1])
    q_result_2 = tf.reshape(q_result_2, [batch_size, -1, 1])

    
    q_result_3 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 0])
    q_result_3 = tf.reshape(q_result_3, [batch_size, -1, 1])

    q_result = tf.concat([q_result_0, q_result_1, q_result_2, q_result_3], axis = -1)

    return q_result   ##  B N 4


def mul_q_point(q_a, q_b, batch_size):

    q_a = tf.reshape(q_a, [batch_size, 1, 4])
    
    q_result_0 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 0])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 1])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 3])
    q_result_0 = tf.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 3])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 2])
    q_result_1 = tf.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 1])
    q_result_2 = tf.reshape(q_result_2, [batch_size, -1, 1])

    
    q_result_3 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 0])
    q_result_3 = tf.reshape(q_result_3, [batch_size, -1, 1])

    q_result = tf.concat([q_result_0, q_result_1, q_result_2, q_result_3], axis = -1)

    return q_result   ##  B N 4



def square_distance(src, dst):

    B = src.get_shape()[0].value
    N = src.get_shape()[1].value
    M = dst.get_shape()[1].value

    for i in range(B):
        ddd = dst[i, :, :]
        sss = src[i, :, :]
        dist_i = -2 * tf.matmul(sss, tf.transpose(ddd, [1, 0]))
        dist_i = tf.expand_dims(dist_i, axis = 0)
        if i == 0:
            dist = dist_i
        else:
            dist = tf.concat([dist, dist_i], axis = 0 )

    dist = dist + tf.reshape(tf.reduce_sum(src ** 2, axis = -1), [B, N, 1])
    dist = dist + tf.reshape(tf.reduce_sum(dst ** 2, axis = -1), [B, 1, M])
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    group_dist, group_idx = tf.nn.top_k(0-sqrdists, nsample)
    return (0-group_dist), group_idx



def warping_layers( xyz1, upsampled_flow):

    return xyz1+upsampled_flow


def cost_volume(warped_xyz, warped_points, f2_xyz, f2_points, nsample, nsample_q, mlp1, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product' ):
    
    
    with tf.variable_scope(scope) as sc:   

        ### FIRST AGGREGATE

        _, idx_q = knn_point(nsample_q, f2_xyz, warped_xyz)
        qi_xyz_grouped = group_point(f2_xyz, idx_q)
        qi_points_grouped = group_point(f2_points, idx_q)

        pi_xyz_expanded = tf.tile(tf.expand_dims(warped_xyz, 2), [1,1,nsample_q,1]) # batch_size, npoints, nsample, 3
        pi_points_expanded = tf.tile(tf.expand_dims(warped_points, 2), [1,1,nsample_q,1]) # batch_size, npoints, nsample, 3
        
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded
        
        pi_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pi_xyz_diff), axis=[-1] , keep_dims=True) + 1e-20 )
    
        pi_xyz_diff_concat = tf.concat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], axis=3)
        
        
        pi_feat_diff = tf.concat(axis=-1, values=[pi_points_expanded, qi_points_grouped])
        pi_feat1_new = tf.concat([pi_xyz_diff_concat, pi_feat_diff], axis=3) # batch_size, npoint*m, nsample, [channel or 1] + 3

        for j, num_out_channel in enumerate(mlp1):
            pi_feat1_new = tf_util.conv2d(pi_feat1_new, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_%d'%(j), bn_decay=bn_decay)


        pi_xyz_encoding = tf_util.conv2d(pi_xyz_diff_concat, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_xyz', bn_decay=bn_decay)

        pi_concat = tf.concat([pi_xyz_encoding, pi_feat1_new], axis = 3)

        for j, num_out_channel in enumerate(mlp2):
            pi_concat = tf_util.conv2d(pi_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_CV_%d'%(j), bn_decay=bn_decay)
        WQ = tf.nn.softmax(pi_concat,dim=2)
            
        pi_feat1_new = WQ * pi_feat1_new
        pi_feat1_new = tf.reduce_sum(pi_feat1_new, axis=[2], keep_dims=False, name='avgpool_diff')#b, n, mlp1[-1]


        ##### SECOND AGGREGATE

        _, idx = knn_point(nsample, warped_xyz, warped_xyz)
        pc_xyz_grouped = group_point(warped_xyz, idx)
        pc_points_grouped = group_point(pi_feat1_new, idx)


        pc_xyz_new = tf.tile( tf.expand_dims (warped_xyz, axis = 2), [1,1,nsample,1] )
        pc_points_new = tf.tile( tf.expand_dims (warped_points, axis = 2), [1,1,nsample,1] )

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new####b , n ,m ,3
        pc_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pc_xyz_diff), axis=3, keep_dims=True) + 1e-20)
        pc_xyz_diff_concat = tf.concat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], axis=3)

        pc_xyz_encoding = tf_util.conv2d(pc_xyz_diff_concat, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_xyz_encoding', bn_decay=bn_decay)

        pc_concat = tf.concat([pc_xyz_encoding, pc_points_new, pc_points_grouped], axis = -1)

        for j, num_out_channel in enumerate(mlp2):
            pc_concat = tf_util.conv2d(pc_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_cost_volume_%d'%(j), bn_decay=bn_decay)
        WP = tf.nn.softmax(pc_concat,dim=2)   #####  b, npoints, nsample, mlp[-1]

        pc_feat1_new = WP * pc_points_grouped

        pc_feat1_new = tf.reduce_sum(pc_feat1_new, axis=[2], keep_dims=False, name='sumpool_diff')#b*n*mlp2[-1]

    return pc_feat1_new



def flow_predictor( points_f1, upsampled_feat, cost_volume, mlp, is_training, bn_decay, scope, bn=True ):

    with tf.variable_scope(scope) as sc:

        if points_f1 == None:
            points_concat = cost_volume

        elif upsampled_feat != None:
            points_concat = tf.concat(axis=-1, values=[ points_f1, cost_volume, upsampled_feat]) # B,ndataset1,nchannel1+nchannel2
        
        elif upsampled_feat == None:
            points_concat = tf.concat(axis=-1, values=[ points_f1, cost_volume]) # B,ndataset1,nchannel1+nchannel2

        points_concat = tf.expand_dims(points_concat, 2)

        for i, num_out_channel in enumerate(mlp):
            points_concat = tf_util.conv2d(points_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_predictor%d'%(i), bn_decay=bn_decay)
        points_concat = tf.squeeze(points_concat,[2])
      
    return points_concat



def sample_and_group(npoint, nsample, xyz, points, use_xyz=True):

    sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    
    if points is None:

        _, idx_q = knn_point(nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx_q)
        grouped_points = group_point(points, idx_q)
        xyz_diff = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) 

        # grouped_xyz, xyz_diff, grouped_points, idx = pointconv_util.grouping(xyz, nsample, xyz, new_xyz)## the method of pointconv
        new_points = tf.concat([xyz_diff, grouped_points] , axis=-1)
    
    else:

        _, idx_q = knn_point(nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx_q)     
        xyz_diff = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) 
   
        # grouped_xyz, xyz_diff, grouped_points, idx = pointconv_util.grouping(points, nsample, xyz, new_xyz)## the method of pointconv
        new_points = tf.concat([xyz_diff, grouped_xyz], axis=-1) # (batch_size, npoint, nample, 3+channel)

    return new_xyz, new_points



def pointnet_sa_module(xyz, points, npoint, nsample, mlp, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', use_xyz=True, use_nchw=False):

    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:

        # Sample and Grouping
        new_xyz, new_points = sample_and_group(npoint, nsample, xyz, points, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2]) 

        for i, num_out_channel in enumerate(mlp):

            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')


        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points


def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', knn=True):

    with tf.variable_scope(scope) as sc:

        _, idx_q = knn_point(nsample, xyz2, xyz1)

        xyz2_grouped = group_point(xyz2, idx_q)
        feat2_grouped = group_point(feat2, idx_q)

        xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint1, nsample, 3

        net = tf.concat([feat2_grouped, xyz_diff], axis=3) # batch_size, npoint1, nsample, channel2+3

        if mlp is None: mlp=[]
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False, name='maxpool') # batch_size, npoint1, mlp[-1]
        elif pooling=='avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False, name='avgpool') # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2) # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new, 2) # batch_size, npoint1, 1, mlp[-1]+channel2
        if mlp2 is None: mlp2=[]
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d'%(i), bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2]) # batch_size, npoint1, mlp2[-1]
        return feat1_new

