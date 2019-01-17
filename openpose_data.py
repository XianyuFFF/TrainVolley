import os
import shutil

openpose_build_path = "/home/fyq/openpose/build"
asset_path = "asset"


def get_openpose_data(video_dir, output_snippets_path):
    video_name = video_dir[video_dir.rfind('/') + 1:].split('.')[0]
    output_json_dir = os.path.join(output_snippets_path, "{}".format(video_name))
    openpose_args = dict(
                video=video_dir,
                write_json=output_json_dir,
                display=0,
                render_pose=0,
                model_pose='BODY_25')

    openpose = '{}/examples/openpose/openpose.bin'.format(openpose_build_path)

    command_line = openpose + ' '
    command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])

    shutil.rmtree(output_json_dir, ignore_errors=True)
    os.makedirs(output_json_dir)
    os.system(command_line)


if __name__ == '__main__':
    video_dir = os.path.join(asset_path, 'SoccerJuggling.avi')
    snippets_path = os.path.join(asset_path, 'openpose_data', 'snippets')

    get_openpose_data(video_dir, snippets_path)