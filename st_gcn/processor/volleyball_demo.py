#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

class Volleyball_Demo(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):
        openpose_work_path = self.arg.openpose_work_path
        output_snippets_dir = '{}/snippets/{}'.format(openpose_work_path, self.arg.video_name)
        output_sequence_dir = '{}/data'.format(openpose_work_path)
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, self.arg.video_name)
        # output_result_dir = self.arg.output_dir
        label_name_path = self.arg.label_name_dir

        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        # pack openpose ouputs
        height, width = self.arg.video_height, self.arg.video_width
        video_info = utils.openpose.json_pack(
            output_snippets_dir, self.arg.video_name, width, height)

        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)

        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)

        if len(video_info['data']) == 0:
            print('Can not find pose estimation results.')
            return
        else:
            print('Pose estimation complete.')

        # parse skeleton data
        pose, _ = utils.video.video_info_parsing(video_info)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        # extract feature
        print('\nNetwork forward...')
        self.model.eval()
        output, feature = self.model.extract_feature(data)
        output = output[0]

        # feature = feature[0]
        # intensity = (feature*feature).sum(dim=0)**0.5
        # intensity = intensity.cpu().detach().numpy()
        # label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        # print('Prediction result: {}'.format(label_name[label]))
        # print('Done.')

        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p] for p in l] for l in label_sequence]


        _, T, V, M = pose.shape
        label_names = []
        main_poses = []
        for t in range(T):
            # if isinstance(main_m, int):
            body_label = label_name_sequence[t // 4][0]
            # else:
            #     body_label = label_name_sequence[t // 4][min(main_m)]
            label_names.append(body_label)

        print(label_names)
        print(main_poses)

        out = {"duration": T, "actions": label_names}

        with open(self.arg.savejson_dir, 'w') as f:
            json.dump(out, f)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        parser.add_argument('--video_name', required=True, type=int, help='video name')
        # region arguments yapf: disable
        parser.add_argument('--video_height', '-vh', required=True, type=int, help='video height')
        parser.add_argument('--video_width', '-vw', required=True, type=int, help='video width')

        parser.add_argument('--openpose_work_path', required=True, type=str)
        parser.add_argument('--label_name_dir', required=True, type=str)

        parser.set_defaults(config='./config/st_gcn/vb14/cross_view/demo.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
