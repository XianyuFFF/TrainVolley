from World import World
from Reconstruct import Camera
from utils.path_parser import get_camera_info_dir
from utils.openpose import json_pack
import argparse
import os

Asset_path = 'asset'

def main():

    parser = argparse.ArgumentParser(
        description='Reconstruct and Analysis for volleyball training')

    parser.add_argument('--camera0-name', '-c0n', type=str, default='cam0')
    parser.add_argument('--camera1-name', '-c1n', type=str, default='cam1')
    parser.add_argument('--c1-video-dir', '-v1', type=str, default=os.path.join(Asset_path, 'video1.mp4'))
    parser.add_argument('--c2-video-dir', '-v2', type=str, default=os.path.join(Asset_path, 'video2.mp4'))
    parser.add_argument('--openpose-build', '-op', type=str,
                        default="/home/fyq/openpose/build/examples/openpose/openpose.bin")
    parser.add_argument('--output_snippets_path', '-osp', type=str,
                        default=os.path.join(Asset_path, 'openpose_data'))

    parser.add_argument('--ball-work-path', '-bwp', type=str,
                        default=os.path.join(Asset_path, 'ball'))

    parser.add_argument('--label_name_dir', type=str, default='st_gcn/resource/**.txt')

    parser.add_argument('--fps', '-f', type=float, default=60.)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    args = parser.parse_args()

    cam0 = Camera(0, args.camera0_name, camera_matrix=None)
    cam0.pack_json_camera(get_camera_info_dir(0, args.camera0_name))

    cam1 = Camera(1, args.camera1_name, camera_matrix=None)
    cam1.pack_json_camera(get_camera_info_dir(1, args.camera1_name))

    cams = [cam0, cam1]
    video_dirs = [args.c1_video_dir, args.c2_video_dir]

    world = World(cams, video_dirs, args.output_snippets_path, args.ball_work_path, args.fps)
    world.detection_and_reconstruct()

    for i, video_dir in enumerate(video_dirs):

        video_name = video_dir[video_dir.rfind('/') + 1:].split('.')[0]
        command_line = './st_gcn/st_gcn.py -vh 1080 -vw 1920 --video_name {} --openpose_work_path {} --label_name_dir {}'
        os.system(command_line)





if __name__ == '__main__':
    main()