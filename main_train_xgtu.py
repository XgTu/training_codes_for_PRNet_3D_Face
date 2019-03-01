import numpy as np
import os, pprint, time
import os, dlib, cv2
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from time import time
import argparse
import ast
pp = pprint.PrettyPrinter()
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

import tensorflow as tf
import tensorlayer as tl
os.environ['CUDA_VISIBLE_DEVICES']='2'
from model import *
from process import *
from random import shuffle

flags = tf.app.flags
flags.DEFINE_string("detector_path", "./Data/net-data/mmod_human_face_detector.dat", "path to face decector")
flags.DEFINE_string("preModel", "./Data/checkpoint/0_model.ckpt", "path to preTrned weights")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("traPath", "./Data/subTrn_list.txt", "path to the input samples .txt")
#flags.DEFINE_string("outputDir", "./TestImages/results/", "path to the output directory")
flags.DEFINE_string("gpu", "2", "id for gpu")
flags.DEFINE_boolean("isDlib", True, "whether to use dlib for detecting face")
flags.DEFINE_boolean("is3d", True, "whether to output 3D face(.obj)")
flags.DEFINE_boolean("isMat", True, "whether to save True,color,triangles as mat for matlab showing")
flags.DEFINE_boolean("isKpt", True, "whether to output key points(.txt)")
flags.DEFINE_boolean("isPose", True, "whether to output estimated pose(.txt)")
flags.DEFINE_boolean("isShow", True, "whether to show the results with opencv(need opencv)")
flags.DEFINE_boolean("isImage", True, "whether to save input image")
flags.DEFINE_boolean("isFront", True, "whether to frontalize vertices(mesh)")
flags.DEFINE_boolean("isDepth", True, "whether to output depth image")
flags.DEFINE_boolean("isTexture", True, "whether to save texture in obj file")
flags.DEFINE_boolean("isMask", True, "whether to set invisible pixels(due to self-occlusion) in texture as 0")
flags.DEFINE_integer("texture_size", 256, "size of texture map, default is 256. need isTexture is True")
flags.DEFINE_integer("resolution_inp", 256, "size of xxx")
flags.DEFINE_integer("resolution_op", 256, "size of xxx")
flags.DEFINE_integer("output_size", 256, "size of the output position maps")
flags.DEFINE_integer("epoch", 50, "maximum epoch of training")
flags.DEFINE_integer("batch_size", 1, "batch size of training & testing")
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(0.0001, global_step, 5000, 0.9, staircase=False)

FLAGS = flags.FLAGS

uv_kpt_ind = np.loadtxt('./Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3
weigh_map = imread('./Data/uv-data/map_xgtu.jpg').astype(np.float32) # 
weigh_map = weigh_map/255. * 16.

def frontalize(vertices):
    canonical_vertices = np.load('./Data/uv-data/canonical_vertices.npy')

    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)
    return front_vertices

def get_landmarks(pos, uv_kpt_ind):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
    return kpt

def generate_uv_coords(resolution_op):
    resolution = resolution_op
    uv_coords = np.meshgrid(range(resolution),range(resolution))
    uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
    uv_coords = np.reshape(uv_coords, [resolution**2,-1]);
    uv_coords = uv_coords[face_ind, :]
    uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
    return uv_coords

def get_vertices(pos, resolution_op):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [resolution_op**2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices

def get_colors_from_texture(texture, resolution_op):
    '''
    Args:
        texture: the texture map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    all_colors = np.reshape(texture, [resolution_op**2, -1])
    colors = all_colors[face_ind, :]
    return colors

def get_colors(image, vertices):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :] # n x 3
    return colors

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)

    uv_coords = generate_uv_coords(FLAGS.resolution_op)

    ##========================= DEFINE MODEL ===========================##
    input_x = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    posGrd = tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3])
    posPret = resfcn256(input_x)
#    loss = tf.nn.l2_loss(posPret-posGrd)/(FLAGS.batch_size*1000)
    loss = tf.reduce_mean(tf.square(posPret-posGrd))
    tvars = [var for var in tf.global_variables() if 'resfcn256'  in var.name]
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
               use_locking=False).minimize(loss, global_step=global_step, var_list=tf.trainable_variables())

    ##========================= RUN TRAINING ===========================##
    iter_counter = 0
    f = open(FLAGS.traPath)
    data_files = f.readlines()
    f.close()
 
#    ff = open('subTrn_list.txt', 'w')
#    for idx in range(len(data_files)):
#        if('posmap.jpg' not in data_files[idx]):
#            ff.write(data_files[idx])
#    ff.close()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(tvars)
    saver.restore(sess, FLAGS.preModel)
    sample = imread(data_files[0][:-2])/255. 
    for epoch in range(FLAGS.epoch):
        shuffle(data_files)
        ## load image data
        batch_idxs = len(data_files) // FLAGS.batch_size

        for idx in range(0, batch_idxs):
            batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            batch_images = [imread(batch_file[:-2])/255. for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32)
            batch_posmaps = [np.load(batch_file.replace('jpg', 'npy')[:-2])/255. for batch_file in batch_files]
            batch_posmaps = np.array(batch_posmaps).astype(np.float32)

            _, posPredict, LOSS, lr = sess.run([train_op, posPret, loss, learning_rate], feed_dict={input_x:batch_images, posGrd:batch_posmaps})
            pos = np.squeeze(posPredict)
            vertices = get_vertices(pos, FLAGS.resolution_op)
#            save_vertices = vertices.copy()
#            save_vertices[:,1] = h - 1 - save_vertices[:,1]
#            sio.savemat(os.path.join('./visualize/prnMat/', str(idx) + '_mesh.mat'), {'vertices': save_vertices, 'triangles': triangles})

            if idx % 200 == 0:
                fp = open(FLAGS.checkpoint_dir + 'print.txt', 'a+w')                 
                print ('[Step:%d|Epoch:%d], lr:%.6f, loss:%.4f' % (idx, epoch, lr, LOSS))
                print>>fp,('[Step:%d|Epoch:%d], lr:%.6f, loss:%.4f' % (idx, epoch, lr, LOSS))
                fp.close()

        saver_path = saver.save(sess, FLAGS.checkpoint_dir + str(epoch) +'_model.ckpt')   
        smpInput = np.expand_dims(sample, axis = 0)
        ppos = sess.run([posPret], feed_dict={input_x:smpInput}) 
        posx = np.squeeze(ppos)
        imsave(os.path.join(FLAGS.checkpoint_dir + str(epoch) +'sample_pos.jpg'), posx)
if __name__ == '__main__':
    tf.app.run()

