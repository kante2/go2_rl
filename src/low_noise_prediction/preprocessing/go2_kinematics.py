"""Go2 foot-frame kinematics from the bundled official URDF (metres/radians).

No Sport foot position/velocity is used to calculate the foot motion. Only
Sport body linear velocity is added to LowState q/dq, gyro and attitude.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

LEGS=('fr','fl','rr','rl')
MODEL_PATH=Path(__file__).resolve().parent/'models/go2_description.urdf'
MODEL_SOURCE='https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf'


def _vector(element,attribute,default='0 0 0'):
    return np.fromstring(element.get(attribute,default) if element is not None else default,sep=' ')


class Go2Kinematics:
    def __init__(self,path=MODEL_PATH):
        self.path=Path(path)
        robot=ET.parse(self.path).getroot()
        joints={joint.get('name'):joint for joint in robot.findall('joint')}
        self.chains={}
        for leg in LEGS:
            chain=[];parent='base'
            for name in ('hip','thigh','calf','foot'):
                joint=joints[f'{leg.upper()}_{name}_joint']
                if joint.find('parent').get('link')!=parent:
                    raise ValueError(f'{leg}: unexpected joint chain')
                parent=joint.find('child').get('link')
                origin=joint.find('origin')
                xyz=_vector(origin,'xyz');rpy=_vector(origin,'rpy')
                axis=_vector(joint.find('axis'),'xyz','1 0 0')
                moving=joint.get('type')=='revolute'
                if moving:
                    axis=axis/np.linalg.norm(axis)
                chain.append((xyz,Rotation.from_euler('xyz',rpy).as_matrix(),axis,moving))
            if sum(item[3] for item in chain)!=3:
                raise ValueError('expected three revolute joints per leg')
            self.chains[leg]=chain

    def position_jacobian(self,leg,q):
        q=np.asarray(q,dtype=float)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():
            raise ValueError('joint angles must be finite (N,3)')
        n=len(q);rot=np.broadcast_to(np.eye(3),(n,3,3)).copy();pos=np.zeros((n,3))
        axes=[];origins=[];j=0
        for xyz,fixed,axis,moving in self.chains[leg]:
            pos+=np.einsum('nij,j->ni',rot,xyz)
            rot=rot@fixed
            if moving:
                axes.append(np.einsum('nij,j->ni',rot,axis));origins.append(pos.copy())
                rot=rot@Rotation.from_rotvec(q[:,j,None]*axis).as_matrix();j+=1
        jacobian=np.stack([np.cross(axis,pos-origin) for axis,origin in zip(axes,origins)],axis=2)
        return pos,jacobian


def world_foot_velocity(position,relative,gyro,rotation,body_velocity,velocity_frame='body'):
    rotating=relative+np.cross(gyro,position)
    motion=rotation.apply(rotating)
    if velocity_frame=='body':
        motion+=rotation.apply(body_velocity)
    elif velocity_frame=='odom':
        motion+=body_velocity
    else:
        raise ValueError('body velocity frame must be body or odom')
    return motion


def add_foot_kinematics(frame,model=None,velocity_frame='body'):
    model=model or Go2Kinematics()
    result=frame.copy()
    quaternion=frame[[f'ls_quaternion_{x}' for x in ('w','x','y','z')]].to_numpy(float)
    if not np.isfinite(quaternion).all() or np.any(np.linalg.norm(quaternion,axis=1)<1e-8):
        raise ValueError('invalid LowState attitude quaternion')
    rotation=Rotation.from_quat(quaternion[:,[1,2,3,0]])
    gyro=frame[[f'ls_gyro_{x}' for x in ('x','y','z')]].to_numpy(float)
    velocity=frame[[f'sp_vel_{x}' for x in ('x','y','z')]].to_numpy(float)
    for leg in LEGS:
        q=frame[[f'ls_q_{leg}_{j}' for j in ('hip','thigh','calf')]].to_numpy(float)
        dq=frame[[f'ls_dq_{leg}_{j}' for j in ('hip','thigh','calf')]].to_numpy(float)
        position,jacobian=model.position_jacobian(leg,q)
        relative=np.einsum('nij,nj->ni',jacobian,dq)
        world=world_foot_velocity(position,relative,gyro,rotation,velocity,velocity_frame)
        for i,axis in enumerate(('x','y','z')):
            result[f'foot_position_body_{leg}_{axis}_m']=position[:,i]
            result[f'foot_velocity_relative_{leg}_{axis}_mps']=relative[:,i]
            result[f'foot_velocity_world_{leg}_{axis}_mps']=world[:,i]
        result[f'foot_speed_world_{leg}_mps']=np.linalg.norm(world,axis=1)
        result[f'foot_downward_speed_{leg}_mps']=np.maximum(-world[:,2],0.)
    return result
