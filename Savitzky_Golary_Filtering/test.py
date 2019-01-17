from Reconstruct import Camera
from utils.path_parser import get_camera_info_dir

cam0 = Camera(0, 'cam0', None)
cam0.pack_json_camera(get_camera_info_dir(cam0.id, cam0.name))

cam1 = Camera(1, 'cam1', None)
cam1.pack_json_camera(get_camera_info_dir(cam1.id, cam1.name))


