import numpy as np
import taichi as ti

from .rigid_link import RigidLink
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


@ti.data_oriented
class SensorLink(RigidLink):
    
    def __init__(
        self,
        resolution,
        fov,
        type,
        scene,
        entity,
        name,
        idx,
        geom_start,
        cell_start,
        vert_start,
        face_start,
        edge_start,
        vgeom_start,
        vvert_start,
        vface_start,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass,
        parent_idx,
        invweight,
        visualize_contact,
    ):
        super().__init__(
            entity=entity,
            name=name,
            idx=idx,
            geom_start=geom_start,
            cell_start=cell_start,
            vert_start=vert_start,
            face_start=face_start,
            edge_start=edge_start,
            vgeom_start=vgeom_start,
            vvert_start=vvert_start,
            vface_start=vface_start,
            pos=pos,
            quat=quat,
            inertial_pos=inertial_pos,
            inertial_quat=inertial_quat,
            inertial_i=inertial_i,
            inertial_mass=inertial_mass,
            parent_idx=parent_idx,
            invweight=invweight,
            visualize_contact=visualize_contact,
        )
        self.scene = scene
        self.cam = self.scene.add_camera(
            res    = resolution,
            pos    = (3.5, 0.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = fov,
            GUI    = True
        )
        self.render_type = {'rgb':False, 'depth':False, 'segmentation':False, 'colorize_seg':False, 'normal':False}
        self.render_type[type] = True

    def render(self):
        radar_quat = self.get_quat()[0].cpu().numpy()
        self.cam.set_pose(
            pos = self.get_pos()[0].cpu().numpy(), 
            lookat = transform_by_quat(np.asarray([1.0,0.0,0.0]), radar_quat),
            up = transform_by_quat(np.asarray([0.0,0.0,1.0]), radar_quat))
        self.cam.render(**self.render_type)