from Player.Player import Player
from openpose_data import get_openpose_data
from reconstruct import find_fundamental_matrix


class World:
    def __init__(self, cams, video_dirs, output_snippets_path, fps):
        super(World, self).__init__()
        self.cams = cams
        self.fundamental_matrix = find_fundamental_matrix(*self.cams)
        self.videos = video_dirs
        self.output_snippets_path = output_snippets_path
        self.player = Player(len(cams))
        self.ball = Player(len(cams))

    def detection_and_reconstruct(self):
        for i, video_dir in enumerate(self.videos):
            get_openpose_data(video_dir, self.output_snippets_path)
            self.player.player_skeletons.load_json_skeleton(video_dir, self.output_snippets_path, i)

        self.player.player_skeletons.reconstruct(self.cams, self.fundamental_matrix)

            # self.ball.loc_location(video_dir, i)

    def analyse(self):
        return {}

    def show_result(self):
        pass