import numpy as np
from .sensor_base import BasicCamera, EquirectangularLidar
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
        self.publisher_pc = ROS.ros_node.create_publisher(ROS.PointCloud2, spec.topic+"_p", 10)
        
        self.name = name
        self.camera = EquirectangularLidar(
            scene = scene, 
            link = link,
            name = name, 
            width = int(spec.resolution[0]), 
            height = int(spec.resolution[1]), 
            hfov = spec.fov[0], 
            vfov = spec.fov[1],
            GUI=False)
        
        self.msg = ROS.PointCloud2()
        self.msg.height = 1
        self.msg.fields = [
            ROS.PointField(name='x', offset=0, datatype=ROS.PointField.FLOAT32, count=1),
            ROS.PointField(name='y', offset=4, datatype=ROS.PointField.FLOAT32, count=1),
            ROS.PointField(name='z', offset=8, datatype=ROS.PointField.FLOAT32, count=1)]
        self.msg.point_step = 12
        self.msg.is_bigendian = False
        self.msg.is_dense = False
        self.msg.header.frame_id = name

    def render(self):
        self.image, self.pts = self.camera.update()
        return self.image, self.pts

    def publish(self):
        self.msg.header.stamp = ROS.ros_node.get_clock().now().to_msg()
        self.msg.width = self.pts.shape[0]
        self.msg.row_step = self.msg.point_step * self.msg.width
        self.msg.data = np.float32(self.pts).tostring()
        self.publisher_pc.publish(self.msg)
        self.publisher.publish(ROS.bridge.cv2_to_imgmsg(self.image, "32FC1"))