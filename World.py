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
        self.analysis_result = None

    def detection_and_reconstruct(self):

        circle_file_dir = os.path.join(Asset_path, "circle.jpg")
        ball_moments = Moments(circle_file_dir)

        for i, video_dir in enumerate(self.videos):
            # TODO if detect file exist check, pass the stage

            # detect stage
            get_openpose_data(video_dir, self.output_snippets_path)
            get_ball_loc_data(video_dir, self.ball_work_path, ball_moments, i)

            # data loading
            self.player.player_skeletons.pack_json_skeleton(video_dir, self.output_snippets_path, i)
            self.ball.trajectory.pack_json_ball(video_dir, self.ball_work_path, i)

        # 3d reconstruct
        self.player.player_skeletons.reconstruct(self.cams, self.fundamental_matrix)
        self.ball.trajectory.reconstruct(self.cams, self.fundamental_matrix)

    def analyse(self, action_sequence):
        actions = merge_action_sequence(action_sequence)
        analysis_results = []
        for i, action in enumerate(actions):
            analysis_results.append(self.analyse_action(action))
        return analysis_results

    def analyse_action(self, action):
        if action.action_name == 'stand':
            return vars(action)

        if action.action_name == 'underhand_server':
            anlysis_result = self.underhand_server_anlysis(action)

        if action.action_name == 'jump_floating_server':
            anlysis_result = self.jump_floating_server_analysis(action)


    def underhand_server_anlysis(self, action):
        start_frame = action.start_frame
        end_frame = action.start_frame

        skeleton3ds = self.player.player_skeletons.sequence_based_time(start_frame, end_frame)
        ball_sequence = self.ball.trajectory.get_ball_loc_sequence(start_frame, end_frame)

        throw_ball_hand = deter_throw_ball_hand(skeleton3ds, ball_sequence)

        begin_time = underhand_server.deter_begin_frame(skeleton3ds, throw_ball_hand)
        beat_time = underhand_server.deter_beat_frame(skeleton3ds, throw_ball_hand)
        recover_time = underhand_server.deter_beat_frame(skeleton3ds, throw_ball_hand)

        ball_land_time = underhand_server.deter_ball_land_frame(ball_sequence, beat_time)

        if throw_ball_hand == 'left':
            throw_ball_shoulder_angle = skeleton3ds[begin_time].left_shoulder_angle
            beat_ball_shoulder_angle = skeleton3ds[beat_time].right_shoulder_angle
            recover_shoulder_angle = skeleton3ds[recover_time].right_shoulder_angle

        else:
            throw_ball_shoulder_angle = skeleton3ds[begin_time].right_shoulder_angle
            beat_ball_shoulder_angle = skeleton3ds[beat_time].left_shoulder_angle
            recover_shoulder_angle = skeleton3ds[recover_time].left_shoulder_angle

        throw_ball_body_ground_angle = skeleton3ds[begin_time].body_ground_angle()
        beat_ball_body_ground_angle = skeleton3ds[beat_time].body_ground_angle()
        recover_body_ground_angle = skeleton3ds[recover_time].body_ground_angle()

        throw_ball_time_cost = (beat_time - begin_time) * (1/self.fps)
        beat_ball_hight = ball_sequence[beat_time].position_3d.center[2]

        player_position = np.average([np.asarray(skeleton3ds[recover_time].pose_3d.left_heel),
                                      np.asarray(skeleton3ds[recover_time].pose_3d.right_heel)],
                                     axis=0)

        ball_land_position = ball_sequence[ball_land_time].position_3d.center
        ball_land_delta = np.asarray(ball_sequence[ball_land_time-2].position_3d.center) - np.asarray(ball_sequence[ball_land_time-1].position_3d.center)
        ball_land_speed = np.sqrt(ball_land_delta.dot(ball_land_delta)) / (1/self.fps)

        ball_beat_position = ball_sequence[beat_time].position.center
        beat_land_delta = np.asarray(ball_land_position) - np.asarray(ball_sequence[ball_land_time-1].position_3d.center)
        ball_average_speed = np.sqrt(beat_land_delta.dot(beat_land_delta)) / ((ball_land_time - beat_time)*1/self.fps)

        ball_beat_delta = np.asarray(ball_sequence[beat_time + 2].position_3d.center) - np.asarray(
            ball_sequence[beat_time + 1].position_3d.center)
        ball_beat_speed = np.sqrt(ball_beat_delta.dot(ball_beat_delta)) / (1 / self.fps)

        # TODO
        hight_over_net = 0

        ball_info = [ball_beat_position, ball_land_position, ball_beat_speed, ball_land_speed, ball_average_speed]

        player_info = [throw_ball_time_cost, beat_ball_hight, player_position, throw_ball_shoulder_angle,
                       beat_ball_shoulder_angle, recover_shoulder_angle, throw_ball_body_ground_angle,
                       beat_ball_body_ground_angle, recover_body_ground_angle]

        self.analysis_result = {
            'ball_info': ball_info,
            'player_info': player_info
        }

    def jump_floating_server_analysis(self, action):
        action_result = {}

        return action_result

    def show_result(self):
        # TODO dosomething_for(self.analysis_result)
        pass