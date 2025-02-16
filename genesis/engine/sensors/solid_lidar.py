import numpy as np
from .sensor_base import BasicCamera, EquirectangularCamera
from genesis.ext.urdfpy import SensorSpec
import genesis.ext.ros2.ros_core as ROS

# @nb.jit()
def lidar_correction(x, hfov, vfov, hres, vres):

    x[1] = -(x[1]/hres - 0.5)
    x[0] = -(x[0]/vres - 0.5)
    zx = 1.0/np.tan(hfov/2)
    zy = 1.0/np.tan(vfov/2)

    rx = np.cos(x[1])*np.cos(x[0])
    ry = np.sin(x[1])*np.cos(x[0])
    rz = np.sin(x[0])

    dot = np.max(np.abs(np.stack((rx, ry, rz), axis=2)), axis=2)
    vector = np.stack((rx, ry, rz), axis=2)/dot.reshape(dot.shape[0],dot.shape[1],1)

    return dot, vector

class SolidLidar():
    def __init__(
            self, 
            scene, 
            link,
            name: str,
            spec: SensorSpec):
        self.publisher = ROS.ros_node.create_publisher(ROS.Image, spec.topic, 10)
        
        self.name = name
        self.camera = BasicCamera(
            scene = scene, 
            link = link,
            name = name, 
            width = int(spec.resolution[0]), 
            height = int(spec.resolution[1]), 
            vfov = spec.fov[1], 
            type = 'depth', 
            GUI=False)
        
        self.cosine_field, self.vector_field = self.correctionFactor()

    def correctionFactor(self):
        return lidar_correction(np.float32(np.mgrid[0:self.vres_raw, 0:self.hres]), self.hfov, self.vfov, self.hres, self.vres_raw)

    def render(self):
        self.image = self.camera.update()
        return self.image

    def publish(self):
        self.publisher.publish(ROS.bridge.cv2_to_imgmsg(self.image, "rgb8"))