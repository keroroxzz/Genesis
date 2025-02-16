import numpy as np
from .sensor_base import BasicCamera, EquirectangularCamera
from genesis.ext.urdfpy import SensorSpec
import genesis.ext.ros2.ros_core as ROS

class Lidar():
    def __init__(
            self, 
            scene, 
            link,
            name: str,
            spec: SensorSpec):
        self.publisher = ROS.ros_node.create_publisher(ROS.Image, spec.topic, 10)
        
        self.name = name
        self.camera = EquirectangularCamera(
            scene = scene, 
            link = link,
            name = name, 
            width = int(spec.resolution[0]), 
            height = int(spec.resolution[1]), 
            vfov = spec.fov[1], 
            type = 'rgb', 
            GUI=False)

    def render(self):
        self.image = self.camera.update()
        return self.image

    def publish(self):
        self.publisher.publish(ROS.bridge.cv2_to_imgmsg(self.image, "rgb8"))