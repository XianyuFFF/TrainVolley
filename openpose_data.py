import os
import shutil

openpose_build_path = "/home/fyq/openpose/build"


def get_openpose_data(video_dir, output_snippets_dir):
    openpose_args = dict(
                video=video_dir,
                write_json=output_snippets_dir,
                display=0,
                render_pose=0,
                model_pose='BODY_25')

    openpose = '{}/examples/openpose/openpose.bin'.format(openpose_build_path)
    command_line = openpose + ' '
    command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])

    shutil.rmtree(output_snippets_dir, ignore_errors=True)
    os.makedirs(output_snippets_dir)
    os.system(command_line)


