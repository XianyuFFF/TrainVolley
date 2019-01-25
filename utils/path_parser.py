# TODO change it to significative

import os

DEFAULT_ASSET_PATH = 'asset'


def detected_ball_json_dir(video_dir, output_path, cam_id):
    video_name = video_dir[video_dir.rfind('/') + 1:].split('.')[0]
    return os.path.join(output_path, video_name + '_' + str(cam_id) + '.json')


def get_camera_info_dir(cam_id, cam_name):
    return os.path.join(DEFAULT_ASSET_PATH, "camera", "{}{}.json".format(cam_id, cam_name))
