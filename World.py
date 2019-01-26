import numpy as np

from Player import Player
from Ball import Ball
from Detect import get_ball_loc_data, get_openpose_data
from Detect import Moments
from Reconstruct.reconstruct import find_fundamental_matrix
import os
from ActionAnylsis import merge_action_sequence
from ActionAnylsis import underhand_server
from ActionAnylsis.underhand_server.determine import deter_throw_ball_hand
from utils.path_parser import *
from utils.helper import get_file_name
from Court import Court, Net
import json


Asset_path = 'asset'


class World:
    def __init__(self, cams, video_dirs, output_snippets_path, ball_work_path, fps):
        super(World, self).__init__()
        self.cams = cams
        self.fundamental_matrix = find_fundamental_matrix(*self.cams)
        self.videos = video_dirs
        self.output_snippets_path = output_snippets_path
        self.player = Player()
        self.ball = Ball()
        self.ball_work_path = ball_work_path
        self.fps = fps
        self.court = Court()

    def detection_and_reconstruct(self):
        for i, video_dir in enumerate(self.videos):
            video_name = get_file_name(video_dir)

            if not os.path.exists(detected_ball_json_dir(video_dir, self.ball_work_path, i)):
                circle_file_dir = os.path.join(Asset_path, "circle.jpg")
                ball_moments = Moments(circle_file_dir)
                get_ball_loc_data(video_dir, self.ball_work_path, ball_moments, i)

            if not os.path.exists(os.path.join(self.output_snippets_path, 'snippets', video_name)):
                get_openpose_data(video_dir, self.output_snippets_path)

        self.player.player_skeletons.pack_json_skeleton(self.videos, self.output_snippets_path, self.cams)
        self.ball.trajectory.pack_json_ball(self.videos, self.ball_work_path)

        # 3d reconstruct
        self.player.player_skeletons.reconstruct(self.cams, self.fundamental_matrix)
        self.ball.trajectory.reconstruct(self.cams, self.fundamental_matrix)

        # visual ball trajectory
        # self.ball.trajectory.show_trajectory()
        self.ball.trajectory.optimize()


    def analyse(self, action_sequence):
        actions = merge_action_sequence(action_sequence)

        analysis_results = []
        for i, action in enumerate(actions):
            analysis_results.append(self.analyse_action(action))
        setattr(self, 'analysis_results', analysis_results)

    def analyse_action(self, action):
        if action.action_name == 'stand':
            return vars(action)

        if action.action_name == 'underhand_server':
            anlysis_result = self.underhand_server_anlysis(action)

        elif action.action_name == 'jump_floating_server':
            anlysis_result = self.jump_floating_server_analysis(action)

        return anlysis_result

    def underhand_server_anlysis(self, action):
        start_frame = action.start_frame
        end_frame = action.end_frame

        skeleton3ds = self.player.player_skeletons.sequence_based_time(start_frame, end_frame)
        ball_sequence = self.ball.trajectory.get_ball_loc_sequence(start_frame, end_frame)

        throw_ball_hand = deter_throw_ball_hand(skeleton3ds, ball_sequence)

        begin_time = underhand_server.deter_begin_frame(skeleton3ds, throw_ball_hand)

        beat_time = underhand_server.deter_beat_frame(skeleton3ds, throw_ball_hand, ball_sequence, begin_time)

        # not really needed
        # recover_time = underhand_server.deter_beat_frame(skeleton3ds, throw_ball_hand, ball_sequence, begin_time)

        ball_land_time = underhand_server.deter_ball_land_frame(ball_sequence, beat_time)

        print(throw_ball_hand)

        if throw_ball_hand == 'left':
            throw_ball_shoulder_angle = skeleton3ds[begin_time].left_shoulder_angle()
            beat_ball_shoulder_angle = skeleton3ds[beat_time].right_shoulder_angle()
            # recover_shoulder_angle = skeleton3ds[recover_time].right_shoulder_angle()
        else:
            throw_ball_shoulder_angle = skeleton3ds[begin_time].right_shoulder_angle()
            beat_ball_shoulder_angle = skeleton3ds[beat_time].left_shoulder_angle()
            # recover_shoulder_angle = skeleton3ds[recover_time].left_shoulder_angle()

        throw_ball_body_ground_angle = skeleton3ds[begin_time].body_ground_angle()
        beat_ball_body_ground_angle = skeleton3ds[beat_time].body_ground_angle()

        # recover_body_ground_angle = skeleton3ds[recover_time].body_ground_angle()

        throw_ball_time_cost = (beat_time - begin_time) * (1 / self.fps)
        throw_ball_hight = ball_sequence[begin_time].position_3d.center[2]
        beat_ball_hight = ball_sequence[beat_time].position_3d.center[2]

        throw_hight = abs(throw_ball_hight - beat_ball_hight)

        # player_position = np.average([np.asarray(skeleton3ds[recover_time].pose_3d.left_heel),
        #                               np.asarray(skeleton3ds[recover_time].pose_3d.right_heel)],
        #                              axis=0)

        fouled = self.player.foul_detection(start_frame, begin_time, beat_time, self.court)

        ball_land_position = ball_sequence[ball_land_time].position_3d.center
        ball_land_delta = np.asarray(ball_sequence[ball_land_time - 2].position_3d.center) - np.asarray(
            ball_sequence[ball_land_time - 1].position_3d.center)
        ball_land_speed = np.sqrt(ball_land_delta.dot(ball_land_delta)) / (1 / self.fps)

        ball_beat_position = ball_sequence[beat_time].position_3d.center
        beat_land_delta = np.asarray(ball_land_position) - np.asarray(ball_beat_position)
        ball_average_speed = np.sqrt(beat_land_delta.dot(beat_land_delta)) / (
                    (ball_land_time - beat_time) * 1 / self.fps)

        ball_beat_delta = np.asarray(ball_sequence[beat_time + 2].position_3d.center) - np.asarray(
            ball_sequence[beat_time + 1].position_3d.center)
        ball_beat_speed = np.sqrt(ball_beat_delta.dot(ball_beat_delta)) / (1 / self.fps)

        ball_fly_time = (ball_land_time - beat_time) * (1/ self.fps)

        # TODO
        over_net_time = 0
        hight_over_net = 0

        # ball_info = [ball_beat_position, ball_land_position, ball_beat_speed, ball_land_speed, ball_average_speed]
        #
        # player_info = [throw_ball_time_cost, beat_ball_hight, player_position, throw_ball_shoulder_angle,
        #                beat_ball_shoulder_angle, recover_shoulder_angle, throw_ball_body_ground_angle,
        #                beat_ball_body_ground_angle, recover_body_ground_angle]

        throw_ball_time_info = {
            "Start_time": start_frame + begin_time,
            "Ball info": {
                "Throw ball time cost": throw_ball_time_cost,
                "Throw ball fly high": throw_hight,
            },
            "Player info": {
                "Throw ball body ground angle": throw_ball_body_ground_angle,
                "Throw ball shoulder angle": throw_ball_shoulder_angle
            }
        }

        beat_ball_info = {
            "Start_time": start_frame + beat_time,
            "Ball info": {
                "Ball beat high": beat_ball_hight,
                "Ball beat speed": ball_beat_speed,
            },
            "Player info": {
                "Beat ball body ground angle": beat_ball_body_ground_angle,
                "Beat ball shoulder angle": beat_ball_shoulder_angle
            }
        }

        over_net_info = {
            "Start_time": start_frame + over_net_time,
            "Ball info": {
                "Over net high": hight_over_net,
            }

        }

        land_ball_info = {
            "Start_time": start_frame + ball_land_time,
            "Ball info": {
                "Ball land speed": ball_land_speed,
                "Ball land position": ball_land_position,
                "Ball average speed": ball_average_speed
            }
        }

        # recover_info = {
        #     "Start_time": start_frame + recover_time,
        #     "Player info": {
        #         "Recover body ground angle": recover_body_ground_angle,
        #         "Recover shoulder angle": recover_shoulder_angle,
        #         "Player position": player_position
        #     }
        # }

        analysis_result = {
            "Start_time": start_frame,
            "End_time": end_frame,
            "Fouled": fouled,
            "BallFlyTime": ball_fly_time,
            "Stages": {
                "Throw ball": throw_ball_time_info,
                "Beat ball": beat_ball_info,
                "Over net": over_net_info,
                # "Recover": recover_info,
                "Land": land_ball_info
            }
        }

        return analysis_result

    def jump_floating_server_analysis(self, action):
        action_result = {}

        return action_result

    def save_demo_result(self, analysis_result):
        video_dir = self.videos[0]
        video_jsa = video_dir.replace('mp4', 'jsa')
        video_jso = video_dir.replace('mp4', 'json')

        # with open(video_jso, 'w') as f:
        #     json.dump(analysis_result, f)

        contents = []
        with open(video_jsa, 'w') as f:

            action_begin = analysis_result['Start_time']
            action_end = analysis_result['End_time']
            stage_num = len(analysis_result['Stages'])

            print(analysis_result['Stages'])

            for stage_name, action_stage in analysis_result['Stages'].items():
                time = int(action_stage['Start_time']) * (1 / self.fps)
                duration = abs(action_begin - action_end) / (stage_num * 1.5) * (1 / self.fps)
                text = str(action_stage)
                contents.append("{start: {}, duration:{}, text: {}\}".format(time, duration, text))
            f.write(str(contents))

        #
        # command_line = 'python3 show_video.py -v {}'.format(video_dir)
        # os.system(command_line)


    # def show_result(self):
    #     video_dir = self.videos[0]
    #     video_jsa = self.videos[0].replace('mp4', 'jsa')
    #
    #     contents = []
    #     with open(video_jsa, 'w') as f:
    #         for i, analysis_result in enumerate(getattr(self, 'analysis_results')):
    #             action_begin = analysis_result['Start_time']
    #             action_end = analysis_result['End_time']
    #             stage_num = len(analysis_result['Stages'])
    #             for j, action_stage in enumerate(analysis_result):
    #                 time = action_stage['Start_time'] * (1 / self.fps)
    #                 duration = abs(action_begin - action_end) / (stage_num * 1.5)
    #                 text = str(action_stage).replace('{', '\n').replace(',', '\n').replace('}', '\n')
    #                 contents.append({"start": time, "duration": duration, "text": text})
    #     f.write(str(contents))
    #
    #     command_line = 'python3 UI/show_video.py -v {}'.format(self.videos[0])
    #     os.system(command_line)