'''
    Single-GPU training code
'''

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import kitti_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='train/test mode')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='PWCLO_Net', help='Model name [default: PWCLO_Net]')

parser.add_argument('--data_root', default='../', help='Path to dataset directory ')
parser.add_argument('--checkpoint_path', default = None, help='Path to the saved checkpoint')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log_train]')
parser.add_argument('--result_dir', default='result', help='result dir [default: result]')

parser.add_argument('--train_list', nargs='+', type=int, default=range(7), help=' List of sequences for training [default: range(7)]')
parser.add_argument('--val_list', nargs='+', type=int, default=range(11), help=' List of sequences for validation [default: range(7, 11)]')
parser.add_argument('--test_list', nargs='+', type=int, default=range(11), help='List of sequences for testing [default: range(11)]')

parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 300]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')##########decay############3
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')


FLAGS = parser.parse_args()

MODE = FLAGS.mode

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data_root
CHECKPOINT_PATH = FLAGS.checkpoint_path
RESULT_PATH = FLAGS.result_dir
TRAIN_LIST = FLAGS.train_list
VAL_LIST = FLAGS.val_list
TEST_LIST = FLAGS.test_list
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
UTIL_FILE = os.path.join(BASE_DIR, 'utils/PWCLO_util.py')
LOG_DIR = FLAGS.log_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (UTIL_FILE, LOG_DIR)) ###SAVE THE UTIL FILE
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 
os.system('cp %s %s' % (__file__, LOG_DIR))
os.system('cp %s %s' % ('kitti_dataset.py', LOG_DIR))

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = kitti_dataset.OdometryDataset(DATA, npoints=NUM_POINT, is_training = True)
TEST_DATASET = kitti_dataset.OdometryDataset(DATA, npoints=NUM_POINT, is_training = False)



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def main(mode = 'train'):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            pointclouds_pl, q_gt, t_gt = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            w_x = tf.Variable(0.0, trainable = True, name = 'w_x')
            w_q = tf.Variable(-2.5, trainable = True, name = 'w_q')

            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")

            # Get model and loss

            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t = MODEL.get_model( pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss( l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)

            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)


        # Init variables

        if CHECKPOINT_PATH != None:
            model_path = CHECKPOINT_PATH
            saver.restore(sess, model_path)
            log_string ("model restored")

        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            log_string ("Initialize model")


        ops = {'pointclouds_pl': pointclouds_pl,            
               'pred_q': l0_q,
               'pred_t': l0_t,
               'is_training_pl': is_training_pl,
               'q_gt': q_gt,
               't_gt': t_gt,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               }
    
        min_eval_error = 1000.0

        if mode == 'train':

            for epoch in range(200, MAX_EPOCH):

                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                train_one_epoch(sess, ops, train_writer, train_list = TRAIN_LIST)


                if epoch % 20 == 0 and epoch <= 100:

                    cur_eval_error = eval_one_epoch(sess, ops, test_list = VAL_LIST)
                    
                    if cur_eval_error < min_eval_error:
                        min_eval_error = cur_eval_error
                        save_dir = os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '_best_model_dir')
                        os.mkdir(save_dir)
                        save_path = saver.save(sess, os.path.join(save_dir, str(cur_eval_error)+"_t_error_model.ckpt"))
                        log_string("Model saved in file: %s" % save_path)



                if epoch % 2 == 0 and epoch > 100:

                    cur_eval_error = eval_one_epoch(sess, ops, test_list = VAL_LIST)
                    
                    if cur_eval_error < min_eval_error:
                        min_eval_error = cur_eval_error
                        save_dir = os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '_best_model_dir')
                        os.mkdir(save_dir)
                        save_path = saver.save(sess, os.path.join(save_dir, str(cur_eval_error)+"_t_error_model.ckpt"))
                        log_string("Model saved in file: %s" % save_path)

        elif mode == 'test':

            if CHECKPOINT_PATH != None:
                eval_one_epoch(sess, ops, test_list = TEST_LIST)
                log_string("Finished! Please check the result directory! ")
            else:
                log_string('Please verify the checkpoint for testing !!!')

def get_batch(dataset, idxs, start_idx, end_idx):

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 3))
    batch_q_gt = np.zeros((bsize, 4))
    batch_t_gt = np.zeros((bsize, 3, 1))

    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)


    for i in range(bsize):

        pc1, pc2, q_gt, t_gt = dataset[idxs[i+start_idx]]########################################
        
        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx]

        batch_q_gt[i, :] = q_gt
        batch_t_gt[i, :, :] = t_gt

    return batch_data, batch_q_gt, batch_t_gt



def train_one_epoch(sess, ops, train_writer, train_list = range(7)):

    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT

    is_training = True

    total_num = 0

    for ii in train_list:

        s = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000]
        e = [4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201]        

        start = s[ii]
        end = e[ii]
         
        total_num += (end - start)
    
    train_idxs = np.arange(total_num)
    
    np.random.shuffle(train_idxs)

    num_batches = len(train_idxs)// BATCH_SIZE

    log_string(str(datetime.now()))

    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, q_gt, t_gt = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['q_gt']: q_gt,
                     ops['t_gt']: t_gt,
                     ops['is_training_pl']: is_training}


        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss']], feed_dict = feed_dict)



        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0 
    EPOCH_CNT += 1

def eval_one_epoch(sess, ops, test_list = range(11)):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    num_eval = 0
    total_t_error = 0
    eval_result = 0

    for ii in test_list:

        s = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000]
        e = [4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201]        

        start = s[ii]
        end = e[ii]
            
        test_idxs = np.arange(start, end)

        num_batches = (end-start+BATCH_SIZE-1) // BATCH_SIZE

        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

        batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
        q_gt = np.zeros([BATCH_SIZE, 4 ])
        t_gt = np.zeros([BATCH_SIZE, 3, 1])
        
        tmp = 0

        for batch_idx in range(num_batches):

            if batch_idx % 100 == 0:
                log_string('---- batch %03d in evaluation ----'%(batch_idx))


            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(end-start, (batch_idx+1) * BATCH_SIZE)
            cur_batch_size = end_idx-start_idx
            
            cur_batch_data, cur_q_gt, cur_t_gt = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

            if cur_batch_size == BATCH_SIZE:
                batch_data = cur_batch_data
                q_gt = cur_q_gt
                t_gt = cur_t_gt
            else:
                batch_data[0:cur_batch_size] = cur_batch_data
                q_gt[0:cur_batch_size] = cur_q_gt
                t_gt[0:cur_batch_size] = cur_t_gt
            
            # ---- INFERENCE BELOW ----

            feed_dict = {ops['pointclouds_pl']: batch_data,
                        ops['q_gt']: q_gt,
                        ops['t_gt']: t_gt,
                        ops['is_training_pl']: is_training}

            
            pred_q, pred_t = sess.run([ops['pred_q'], ops['pred_t']], feed_dict=feed_dict)

            for n0 in range(cur_batch_size):

                if BATCH_SIZE != 1:
                    q_one_batch = pred_q[n0:n0+1, :]
                    t_one_batch = pred_t[n0:n0+1, :]
                else:
                    q_one_batch = pred_q
                    t_one_batch = pred_t

                qq = np.reshape(q_one_batch, [4])
                tt = np.reshape(t_one_batch, [3, 1])
                
                RR = quat2mat(qq)

                filler = np.array([0.0, 0.0, 0.0, 1.0])
                filler = np.expand_dims(filler, axis = 0)   ##1*4

                TT = np.concatenate([np.concatenate([RR, tt], axis=-1), filler], axis=0)

                if tmp == 0:

                    T_final = TT ### 4 4 
                    T = T_final[ :3, : ]####  3 4
                    T = T.reshape(1,1,12)
                    tmp += 1

                else:
                    T_final = np.matmul(T_final, TT)                 
                    T_current = T_final[ :3, : ]
                    T_current = T_current.reshape(1,1,12)
                    T = np.append(T, T_current, axis=0)
            
        T = T.reshape(-1, 12)

        fname_txt = os.path.join(LOG_DIR, str(ii).zfill(2) + '_pred.txt')
        result_dir = RESULT_PATH

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        np.savetxt(fname_txt, T, fmt='%.08f')
        os.system('cp %s %s' % (fname_txt, result_dir)) ###  SAVE THE txt FILE

        result_f = os.popen("python ./evaluation.py --result_dir " + result_dir + " --eva_seqs " + str(ii).zfill(2) + "_pred", "r")

        for line in result_f.readlines():
            
            log_string(line)

            if ('seq' in line):
            
                cur_t_error = float(line.strip().split(' ')[-3])
                total_t_error += cur_t_error
                num_eval += 1

    eval_result = total_t_error/num_eval

    return eval_result


def quat2mat(q):
    
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.
    References
    '''
    
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])




if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    main(MODE)
    LOG_FOUT.close()
