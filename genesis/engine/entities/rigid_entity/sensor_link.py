import numpy as np
import taichi as ti

from .rigid_link import RigidLink
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from genesis.engine.sensors import *

@ti.data_oriented
class SensorLink(RigidLink):
    
    def __init__(
        self,
        spec,
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
        self.sensor = eval(spec.type)(scene, self, name, spec)

    def render(self):
        self.sensor.render()

    def publish(self):
        self.sensor.publish()