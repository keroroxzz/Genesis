import numpy as np
import numba
from abc import ABC, abstractmethod
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, pos_lookat_up_to_T

class BasicCamera:
    render_index = {'rgb':0, 'depth':1, 'segmentation':2, 'normal':3}
    def __init__(
            self, 
            scene, 
            link,
            name: str, 
            width: int, 
            height: int, 
            fov: float, 
            type: str, 
            GUI=False, 
            lookat=np.asarray([1.0,0.0,0.0]),
            up=np.asarray([0.0,0.0,1.0])):
        self.scene = scene
        self.link = link
        self.name = name
        self.lookat = lookat
        self.up = up
        self.camera = self.scene.add_camera(
            res    = (width, height),
            fov    = fov,
            GUI    = GUI
        )
        self.render_type_args = {'rgb':False, 'depth':False, 'segmentation':False, 'normal':False}
        self.render_type_args[type] = True
        self.render_type = type

    def update(self):
        pos = self.link.get_pos()[0].cpu().numpy()
        quat = self.link.get_quat()[0].cpu().numpy()
        self.camera.set_pose(
            pos = pos, 
            lookat = transform_by_quat(self.lookat, quat),
            up = transform_by_quat(self.up, quat))
        render_imgs = self.camera.render(**self.render_type_args)
        return render_imgs[self.render_index[self.render_type]]

# @numba.jit(cache=True, nogil=True)
def generateLidarPoints(hres, vres, hfov, vfov):
    # generate the 3d points in the camera frame, [n, 4]
    x = np.mgrid[0:hres, 0:vres].astype(np.float32)
    x[0] = (x[0]/hres - 0.5)*hfov
    x[1] = (x[1]/vres - 0.5)*vfov

    rx = np.cos(x[0])*np.cos(x[1])
    ry = np.sin(x[0])*np.cos(x[1])
    rz = np.sin(x[1])

    return np.stack((rx, ry, rz, np.ones(rx.shape)), axis=2).reshape(-1,4)

# @numba.jit(cache=True, nogil=True)
def projectLidarPoints(lidarPoints, extrinsic, intrinsic, width, height):
    # project the 3d points to the image plane, and return the 2d points within image region, [n, 2]
    w2i = np.matmul(intrinsic, extrinsic)
    pts = np.matmul(w2i, lidarPoints.T).T
    pts = pts[:,:2]/pts[:,2:3]

    vec = np.matmul(extrinsic, lidarPoints.T).T
    vec = lidarPoints[:,:3] / vec[:,2:3]

    # filter out points out of the boundary of the camera
    is_x_inbounds = np.logical_and(pts[:,0] >= 0.0, pts[:,0] <= width)
    is_y_inbounds = np.logical_and(pts[:,1] >= 0.0, pts[:,1] <= height)
    is_inbounds = np.logical_and(is_x_inbounds, is_y_inbounds)
    pts = pts[is_inbounds]
    vector = vec[is_inbounds]
    return pts, vector

# @numba.jit(cache=True, nogil=True)
def getFractalImagePoints(imagePts):
    fract, ints = np.modf(imagePts)
    return fract, ints.astype(np.int64)

@numba.jit(cache=True, nogil=True, nopython=True)
def img2pts(img, pixelPts, fract, vecs):
    # sample the image by linear interpolation at the image points, [n, 2]
    ret = np.zeros((pixelPts.shape[0],1))
    for i in range(pixelPts.shape[0]):
        x, y = pixelPts[i]
        rx, ry = fract[i]
        a = img[y,   x]   * (1.0-rx) + img[y+1, x]   * rx
        b = img[y,   x+1] * (1.0-rx) + img[y+1, x+1] * rx
        ret[i,0] = (a * (1.0-ry) + b * ry)
    return ret*vecs

class EquirectangularLidar:
    CAMERA_COUNT = 4
    render_index = 1 # depth
    render_type_args = {'rgb':True, 'depth':True, 'segmentation':False, 'normal':False}

    def __init__(
            self, 
            scene, 
            link,
            name: str, 
            width: int, 
            height: int, 
            vfov: float, 
            hfov: float, 
            GUI=False):
        self.scene = scene
        self.link = link
        self.name = name
        self.head = np.asarray([1.0,0.0,0.0])
        self.up = np.asarray([0.0,0.0,1.0])

        # calculate the image size & fov of each camera
        hfov_rad = np.deg2rad(hfov)
        vfov_rad = np.deg2rad(vfov)

        hfovPerCam = hfov_rad/self.CAMERA_COUNT
        widthPerCam = int(width/self.CAMERA_COUNT)
        focal = float(widthPerCam)/np.tan(hfovPerCam/2.0)/2.0
        heightPerCam = (2.1*np.sqrt(np.square(focal)+np.square(widthPerCam/2.0))*np.tan(vfov_rad/2.0)).astype(np.int32)
        camera_vfov = np.rad2deg(2.0*np.arctan(heightPerCam/2.0/focal))

        self.cameras = []
        self.lookats = []

        # pre-cook the lidar points
        lidarPoints = generateLidarPoints(width, height, hfov_rad, vfov_rad)

        self.samplePixels = []
        self.sampleFracts = []
        self.lidarVectors = []
        for i in range(self.CAMERA_COUNT):
            lookat = np.asarray([np.cos(i*hfovPerCam), np.sin(i*hfovPerCam),0.0])
            camera = self.scene.add_camera(
                    res    = (widthPerCam, heightPerCam),
                    fov    = camera_vfov,
                    GUI    = GUI,
                )
            camera._transform = pos_lookat_up_to_T(np.array([0.0,0.0,0.0]), lookat, self.up)
            self.lookats.append(lookat)
            self.cameras.append(camera)

            # project the lidar points to the image plane
            pts, vec = projectLidarPoints(lidarPoints, camera.extrinsics[:3,:], camera.intrinsics, widthPerCam, heightPerCam)
            fract, pixelPts = getFractalImagePoints(pts)
            pixelPts[:,0] += (self.CAMERA_COUNT-1-i)*widthPerCam

            self.samplePixels.append(pixelPts)
            self.sampleFracts.append(fract)
            self.lidarVectors.append(vec)
        
        # concatenate the samples points
        self.samplePixels = np.concatenate(self.samplePixels, axis=0)
        self.sampleFracts = np.concatenate(self.sampleFracts, axis=0)
        self.lidarVectors = np.concatenate(self.lidarVectors, axis=0)
            

    def update(self):
        pos = self.link.get_pos()[0].cpu().numpy()
        quat = self.link.get_quat()[0].cpu().numpy()
        up = transform_by_quat(self.up, quat)
        imgs = []
        for i in range(self.CAMERA_COUNT):
            self.cameras[i].set_pose(
                pos = pos, 
                lookat = transform_by_quat(self.lookats[i], quat)+pos,
                up = up)
            img = self.cameras[i].render(**self.render_type_args)[self.render_index]
            imgs.append(img)
        full_img = np.concatenate(imgs[::-1], axis=1)
        return img2pts(full_img, self.samplePixels, self.sampleFracts, self.lidarVectors)
