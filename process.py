import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
import dlib

detector_path = os.path.join('./Data/net-data/mmod_human_face_detector.dat')
face_detector = dlib.cnn_face_detection_model_v1(detector_path)

def cropImg(image, image_info = None, resolution_inp=256):
    ''' process image with crop operation.
    Args:
        input: (h,w,3) array or str(image path). image value range:1~255. 
        image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

    Returns:
    pos: the 3D position map. (256, 256, 3).
    '''
    if image.ndim < 3:
        image = np.tile(image[:,:,np.newaxis], [1,1,3])

    if image_info is not None:
        if np.max(image_info.shape) > 4: # key points to get bounding box
            kpt = image_info
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
            top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
        else:  # bounding box
            bbox = image_info
            left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size*1.6)
    else:
        detected_faces = face_detector(image)
        if len(detected_faces) == 0:
            print('warning: no detected face')
            return None

        d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
        left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
        size = int(old_size*1.58)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
        
    image = image/255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))

    return cropped_image, tform


import numpy as np
from utils.render import vis_of_vertices, render_texture
from scipy import ndimage

def get_visibility(vertices, triangles, h, w):
    triangles = triangles.T
    vertices_vis = vis_of_vertices(vertices.T, triangles, h, w)
    vertices_vis = vertices_vis.astype(bool)
    for k in range(2):
        tri_vis = vertices_vis[triangles[0,:]] | vertices_vis[triangles[1,:]] | vertices_vis[triangles[2,:]]
        ind = triangles[:, tri_vis]
        vertices_vis[ind] = True
    # for k in range(2):
    #     tri_vis = vertices_vis[triangles[0,:]] & vertices_vis[triangles[1,:]] & vertices_vis[triangles[2,:]]
    #     ind = triangles[:, tri_vis]
    #     vertices_vis[ind] = True
    vertices_vis = vertices_vis.astype(np.float32)  #1 for visible and 0 for non-visible
    return vertices_vis

def get_uv_mask(vertices_vis, triangles, uv_coords, h, w, resolution):
    triangles = triangles.T
    vertices_vis = vertices_vis.astype(np.float32)
    uv_mask = render_texture(uv_coords.T, vertices_vis[np.newaxis, :], triangles, resolution, resolution, 1)
    uv_mask = np.squeeze(uv_mask > 0)
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = uv_mask.astype(np.float32)

    return np.squeeze(uv_mask)

def get_depth_image(vertices, triangles, h, w, isShow = False):
    z = vertices[:, 2:]
    if isShow:
        z = z/max(z)
    depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1)
    return np.squeeze(depth_image)