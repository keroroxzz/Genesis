import numpy as np
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


def generateLidarPoints(hres, vres, hfov, vfov):
    x = np.mgrid[0:hres, 0:vres]
    x[1] = -(x[1]/hres - 0.5)*hfov
    x[0] = -(x[0]/vres - 0.5)*vfov

    rx = np.cos(x[1])*np.cos(x[0])
    ry = np.sin(x[1])*np.cos(x[0])
    rz = np.sin(x[0])

    return np.stack((-ry, -rz, rx, np.ones(rx.shape)), axis=2).reshape(-1,4)

def projectLidarPoints(lidarPoints, extrinsic, intrinsic, width, height):

    res = np.matmul(extrinsic, lidarPoints.T).T
    res = res/res[:,3:4]

    # filter out points behind the camera
    is_z_inbounds = res[:,2] > 0.0
    res = res[is_z_inbounds]

    res = np.matmul(intrinsic, res[:,:3].T).T
    res = res[:,:2]/res[:,2:3]

    # filter out points out of the FoV of the camera
    is_x_inbounds = np.logical_and(res[:,0] > -width/2.0, res[:,0] < width/2.0)
    is_y_inbounds = np.logical_and(res[:,1] > -height/2.0, res[:,1] < height/2.0)
    is_inbounds = np.logical_and(is_x_inbounds, is_y_inbounds)
    res = res[is_inbounds]
    return res

def projectLidarPointsToImage(lidarPoints, camera, ihres, ivres):
    expanded_intrinsic = np.zeros((4,4))
    expanded_intrinsic[:2,:2] = camera.intrinsics[:2,:2]
    expanded_intrinsic[3,2] = camera.intrinsics[2,2]
    expanded_intrinsic[:2,3] = camera.intrinsics[:2,2]
    projectedPoints = projectLidarPoints(lidarPoints, camera.extrinsics, camera.intrinsics, ivres, ihres)
    return projectedPoints[:,:2].astype(np.int64)

def lidar_correction(x, hfov, vfov, hres, vres):

    x[1] = -(x[1]/hres - 0.5)*hfov
    x[0] = -(x[0]/vres - 0.5)*vfov

    rx = np.cos(x[1])*np.cos(x[0])
    ry = np.sin(x[1])*np.cos(x[0])
    rz = np.sin(x[0])

    dot = np.max(np.abs(np.stack((rx, ry, rz), axis=2)), axis=2)
    vector = np.stack((rx, ry, rz), axis=2)/dot.reshape(dot.shape[0],dot.shape[1],1)

    return dot, vector

class EquirectangularCamera:
    CAMERA_COUNT = 4
    render_index = {'rgb':0, 'depth':1, 'segmentation':2, 'normal':3}
    def __init__(
            self, 
            scene, 
            link,
            name: str, 
            width: int, 
            height: int, 
            vfov: float, 
            type: str, 
            GUI=False):
        self.scene = scene
        self.link = link
        self.name = name
        self.head = np.asarray([1.0,0.0,0.0])
        self.up = np.asarray([0.0,0.0,1.0])

        f = float(height)/np.tan(np.deg2rad(vfov)/2.0)/2.0
        widthPerCam = int(width/self.CAMERA_COUNT)
        hfovPerCam = np.arctan(widthPerCam/f/2.0)*2.0
        self.cameras = []
        self.lookats = []
        self.samplePixels = []

        lidarPoints = generateLidarPoints(width//3, height//5, hfovPerCam*self.CAMERA_COUNT, vfov/2.0)#[:,:,[0,1,3]]

        for i in range(self.CAMERA_COUNT):
            lookat = np.asarray([np.cos(i*hfovPerCam), np.sin(i*hfovPerCam),0.0])
            camera = self.scene.add_camera(
                    res    = (widthPerCam, height),
                    fov    = vfov,
                    GUI    = GUI,
                )
            camera._transform = pos_lookat_up_to_T(np.array([0.0,0.0,0.0]), lookat, self.up)

            self.samplePixels.append(projectLidarPointsToImage(lidarPoints, camera, widthPerCam, height))
            self.lookats.append(lookat)
            self.cameras.append(camera)
        self.render_type_args = {'rgb':False, 'depth':False, 'segmentation':False, 'normal':False}
        self.render_type_args[type] = True
        self.render_type = type

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
            img = self.cameras[i].render(**self.render_type_args)[self.render_index[self.render_type]]
            img[self.samplePixels[i][:,0], self.samplePixels[i][:,1]] = 255
            imgs.append(img)
        return np.concatenate(imgs[::-1], axis=1)