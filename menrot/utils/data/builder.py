import os
import math
import json
import numpy as np
import pandas as pd
from typing import Optional

from shaperenderer import utils 
from shaperenderer.geometry import (
    ShapeString,
    MetzlerShape,
    Quadrant,
    Plane
)
from shaperenderer.renderer import (
    Camera,
    Renderer,
    Object3D
)

__all__ = [
    "MenrotSymbolicBuilder",
    "MenrotCognitiveBuilder",
    "MenrotRendererBuilder"
]

class _Builder:
    def __init__(self, root_dir, task, split='train'):
        self.root_dir = root_dir
        self.task = task           # e.g. 'cognitive', 'symbolic' or 'renderer'
        self.split = split         # e.g. 'train' or 'val'
        
        self.task_dir = os.path.join(root_dir, f'menrot_{task}') 
        self.data_dir = os.path.join(self.task_dir, split)   
           
        if task in ['cognitive', 'symbolic']: 
            csv_path = os.path.join(os.path.dirname(__file__), "configs", f"shapes-{split}.csv")
            self._prepare_directories()
            self.unique_shapes = list(map(ShapeString, pd.read_csv(
                csv_path,
                usecols=["shape_string"]
            ).squeeze()))
            self.metadata_dict = None
        else:
            csv_path = os.path.join(os.path.dirname(__file__), "configs", f"objects_shape-{split}.csv")
            self.unique_objects = list(map(ShapeString, pd.read_csv(
                csv_path,
                usecols=["shape_string"]
            ).squeeze()))  
            

        #-- renderer utilities 
        self.camera = Camera()
        self.renderer = Renderer(
            imgsize=(128, 128),
            dpi=100,
            bgcolor="white",
            format="png"
        );
        self.object_params = {
            "facecolor":  "white",
            "edgecolor": "black",
            "edgewidth": 0.8
        }

    def _prepare_directories(self):
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'imag'), exist_ok=True)

    def _get_quadrant(self,phi):
        if 45 < phi <= 135:
            return 1
        elif 135 < phi <= 225:
            return 2
        elif 225 < phi <= 315:
            return 3
        else:
            return 0

    def _get_quadrant_shift(self, phi_1, phi_2, is_mirror=False):
        qdrt_1 = self._get_quadrant(phi_1)
        qdrt_2 = self._get_quadrant(phi_2)
        
        if qdrt_1 == qdrt_2:
            return 0 if is_mirror == False else 4
        elif qdrt_1 == (qdrt_2 + 1) % 4: 
            return 1 if is_mirror == False else 6 #90 or -90 for mirror
        elif qdrt_1 == (qdrt_2 - 1) % 4: 
            return 2 if is_mirror == False else 5 #-90 or 90 for mirror
        else:
            return 3 if is_mirror == False else 7 #180 

    def _get_endpoint(self, shape_string, phi, theta=-25, rho=25, mirror=False):
        if mirror:
            shape_string = shape_string.reflect(Plane.YZ)
        
        if 45 < phi <= 135: #135:
            phi = 90
        elif 135 < phi <= 225: #elif 135 < phi <= 247.5:
            phi = 180
        elif 225 < phi <= 315 :
            phi = 270
        else:
            phi = 0
            
        elevation = math.radians(theta)
        azimuth = math.radians(phi)
        radius = np.array([0., 0., rho])

        rotation = utils.yrotation(azimuth) @ utils.xrotation(elevation)
        eye_pos = (rotation @ utils.homogenize(radius))[:3]
        
        end_1 = np.array(MetzlerShape(shape_string).centers[0])
        end_2 = np.array(MetzlerShape(shape_string).centers[-1])
        
        dist_1 =np.linalg.norm(end_1 - eye_pos)
        dist_2 =np.linalg.norm(end_2 - eye_pos)
            
        if dist_2 <= dist_1:
            return shape_string.reverse() 
        else:
            return shape_string
    
    def _get_skeleton(self, shape_string, phi):
        if 45 < phi <= 135: #135:
            skeleton = shape_string.change_quadrant(Quadrant.II)
        elif 135 < phi <= 225: #elif 135 < phi <= 247.5:
            skeleton = shape_string.change_quadrant(Quadrant.III)
        elif 225 < phi <= 315 :
            skeleton = shape_string.change_quadrant(Quadrant.IV)
        else:
            skeleton = shape_string.change_quadrant(Quadrant.I)
        return skeleton
    
    def save_figure(self, shape_string, file_name, phi, theta=-25):
        self.camera.setSphericalPosition(
            r=25,
            theta=theta,
            phi=phi
        )
         
        self.renderer.render(
            Object3D(
                shape=MetzlerShape(shape_string),
                **self.object_params
            ), 
            self.camera
        )

        self.renderer.save_figure_to_file(file_name)

    def save_metadata(self):
        metadata_df = pd.DataFrame(self.metadata_dict)
        metadata_df.to_csv(os.path.join(self.data_dir, 'metadata.csv'), index=True)

    def run(self):
        """Must be implemented by subclass."""
        raise NotImplementedError
    

class MenrotCognitiveBuilder(_Builder):      
    def __init__(self, root_dir, split='train', repeat=4):
        super().__init__(root_dir=root_dir, task='cognitive', split=split)

        self.rng_angle = np.random.default_rng(seed=42)
        
        self.is_mirror_task = np.concatenate((
            np.repeat(False, repeat), 
            np.repeat(True , repeat)
        ))

        self.metadata_dict = {
            'id': [],
            'shape_string': [], 
            'skeleton_1': [], 
            'skeleton_2': [], 
            'is_mirror': [], 
            'delta_phi': [], 
            'rot_qdrt': [], 
            'phi_1': [], 
            'phi_2': []
        }

    def update_metadata(self, id_data, shape_string, skeleton_1, skeleton_2, is_mirror, delta_phi, rot_qdrt, phi_1, phi_2):
        self.metadata_dict['id'].append(id_data)
        self.metadata_dict['shape_string'].append(str(shape_string))
        self.metadata_dict['skeleton_1'].append(str(skeleton_1))
        self.metadata_dict['skeleton_2'].append(str(skeleton_2))
        self.metadata_dict['is_mirror'].append(bool(is_mirror))
        self.metadata_dict['delta_phi'].append(delta_phi)
        self.metadata_dict['rot_qdrt'].append(rot_qdrt)
        self.metadata_dict['phi_1'].append(phi_1)
        self.metadata_dict['phi_2'].append(phi_2)


    def run(self):
        for id_exp, shape_string_ in enumerate(self.unique_shapes): 
            print(f'Building: {(id_exp+1)/len(self.unique_shapes) * 100:02.0f}%', end='\r')
            for dphi in [0, 60, 120, 180]:     
                id_task = 0
                for is_mirror in self.is_mirror_task:  
                    phi_1 =  self.rng_angle.uniform(low=0, high=360) % 360
                    phi_2 = (phi_1 + [dphi, -dphi][np.random.randint(0,2)]) % 360
                    
                    rot_qdrt = self._get_quadrant_shift(phi_1=phi_1, phi_2=phi_2, is_mirror=is_mirror)
                    if is_mirror:
                        phi_2 = -phi_2 % 360
                         
                    shape_string_1 = self._get_endpoint(phi=phi_1, shape_string=shape_string_)
                    shape_string_2 = self._get_endpoint(phi=phi_2, shape_string=shape_string_, mirror=is_mirror)
                
                    skeleton_1 = self._get_skeleton(shape_string_1, phi_1)
                    skeleton_2 = self._get_skeleton(shape_string_2, phi_2)
                                    
                    data_id = f'{id_exp:03d}_T{id_task:02d}_D{dphi:03d}_M{int(is_mirror)}'

                    self.save_figure(
                        shape_string=shape_string_1, 
                        phi=phi_1,
                        file_name=os.path.join(os.path.join(self.data_dir, 'imag'), f'{data_id}_{1}.png')
                    )  
                    self.save_figure(
                        shape_string=shape_string_2, 
                        phi=phi_2,
                        file_name=os.path.join(os.path.join(self.data_dir, 'imag'), f'{data_id}_{2}.png')
                    )  

                    
                    self.update_metadata(data_id, shape_string_, skeleton_1, skeleton_2, is_mirror, dphi, rot_qdrt, phi_1, phi_2)
                    id_task += 1

        self.save_metadata()           
        print(f'Data {self.data_dir} ----> DONE')

class MenrotSymbolicBuilder(_Builder):
    def __init__(self, root_dir, split='train'):
        super().__init__(root_dir=root_dir, task='symbolic', split=split)

        self.rng_angle = np.random.default_rng(seed=42)

        self.metadata_dict = {
            'id': [], 
            'shape_string': [], 
            'skeleton': [], 
            'encoding': [], 
            'phi_angle': [],
            'qdrt': []
        }
    
    def update_metadata(self, id_data, shape_string, skeleton, encoding, phi_angle, qdrt):
        self.metadata_dict['id'].append(id_data)
        self.metadata_dict['shape_string'].append(str(shape_string))
        self.metadata_dict['skeleton'].append(str(skeleton))
        self.metadata_dict['encoding'].append(encoding)
        self.metadata_dict['phi_angle'].append(phi_angle)
        self.metadata_dict['qdrt'].append(qdrt)

    def run(self):
        data_id = 0 
        for i, shape_string_ in enumerate(self.unique_shapes):
            print(f'Building: {(i+1)/len(self.unique_shapes) * 100:02.0f}%', end='\r')
            qdrt1 = self.rng_angle.uniform(low=-45, high=45, size=42)
            qdrt2 = self.rng_angle.uniform(low=45, high=135, size=42)
            qdrt3 = self.rng_angle.uniform(low=135, high=225, size=42)
            qdrt4 = self.rng_angle.uniform(low=225, high=315, size=42)
            possible_angles = np.concatenate((qdrt1, qdrt2, qdrt3, qdrt4))
            for phi in possible_angles :
                phi = phi % 360
                
                qdrt = self._get_quadrant(phi)
                shape_string = self._get_endpoint(shape_string=shape_string_, phi=phi) 
                skeleton = self._get_skeleton(shape_string, phi)

                self.save_figure(
                    shape_string=shape_string, 
                    phi=phi,
                    file_name=os.path.join(os.path.join(self.data_dir, 'imag'), f'{data_id}.png')
                )
                self.update_metadata(data_id, shape_string_, skeleton, skeleton.encode(), phi, qdrt)
                data_id += 1
        self.save_metadata()    
        print(f'\nData {self.data_dir} ----> DONE')

class MenrotRendererBuilder(_Builder):
    def __init__(
        self, 
        root_dir: str, 
        split: str,
        num_views: int=250,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(root_dir=root_dir, task='renderer', split=split)
        
        self.num_scenes = len(self.unique_objects)
        self.num_imgs_per_scene = num_views
        
        self.rng = np.random.default_rng(seed=seed)  

    def run(self):
        for idx in range(self.num_scenes):
            print(f' Generating shape {1+idx:03d}/{self.num_scenes}')
            os.makedirs(os.path.join(self.data_dir, f'{self.unique_objects[idx].__repr__()}'), exist_ok=True)

            metadata = {}
            for view_idx in range(self.num_imgs_per_scene):
                shape = self.unique_objects[idx]
                theta, phi = utils.sample_on_sphere(low=0.05, high=0.95, size=(1, 1), rng=self.rng)
                skeleton = self._get_skeleton(shape, theta=theta, phi=phi)

                metadata.update({str(f'{view_idx:05d}'): {"azimuth": float(phi[0]), "elevation": float(-theta[0]), "skeleton": skeleton.__repr__()}})
                
                _file_name = str(f'{view_idx:05d}')  
                self.save_figure(
                    shape_string=shape, 
                    theta=theta,
                    phi=phi,
                    file_name=os.path.join(os.path.join(self.data_dir, shape.__repr__()), f'{_file_name}.png')
                )
                                
            file_path = os.path.join(os.path.join(self.data_dir, f'{self.unique_objects[idx].__repr__()}'), "render_params.json" )
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        print(f'Data {self.data_dir} ----> DONE')
        
    def _get_skeleton(self, shape_string, phi, theta):
        shape_string = self._get_endpoint(shape_string, theta=theta, phi=phi)
        phi = phi % 360
        top_map = {'U': 'F', 'D': 'B','F': 'D','B': 'U','R': 'R','L': 'L'}
        bot_map = {'U': 'B', 'D': 'F','F': 'U','B': 'D','R': 'R','L': 'L'}
        
        #--- skeleton part phi ---#
        if 45 < phi <= 135: #135:
            skeleton = shape_string.change_quadrant(Quadrant.II)
        elif 135 < phi <= 225: #elif 135 < phi <= 247.5:
            skeleton = shape_string.change_quadrant(Quadrant.III)
        elif 225 < phi <= 315 :
            skeleton = shape_string.change_quadrant(Quadrant.IV)
        else:
            skeleton = shape_string.change_quadrant(Quadrant.I)
            
        #--- skeleton part theta---#
        if 45 < theta: 
            skeleton =  ShapeString(''.join([bot_map[char.upper()] for char in skeleton])) 
        elif theta <= -45: #elif 135 < phi <= 247.5:
            skeleton =  ShapeString(''.join([top_map[char.upper()] for char in skeleton]))
        else:
            skeleton =  skeleton
            
        return skeleton #.encode()
        

    def __len__(self):
        return len(self.unique_objects) * self.num_views
    