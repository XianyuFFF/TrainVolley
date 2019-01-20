import cv2
from Detect import BgsModel
import os
from utils.path_parser import detected_ball_json_dir
from Detect.parabola_filter import filter_locs


def get_ball_loc_data(video_dir, ball_work_path, ball_moments, cam_id, min_ball_size=4, max_ball_size=30):

    bgs = BgsModel()
    cap = cv2.VideoCapture(video_dir)

    locs = []
    num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tmp_image_dir = os.path.join(ball_work_path, 'tmp', "{}.png".format(num))
        cv2.imwrite(tmp_image_dir, bgs(frame))
        frame_different_region = ball_moments.find_loc(cv2.imread(tmp_image_dir))

        # frame_different_center = frame_different_region[:2]
        # frame_different_size = frame_different_region[2:]
        locs.append(frame_different_region)
        num += 1

    final_loc = filter_locs(locs, min_ball_size, max_ball_size)

    output_json_dir = detected_ball_json_dir(video_dir, ball_work_path, cam_id)
    with open(output_json_dir, 'w') as f:
        f.write(final_loc)
