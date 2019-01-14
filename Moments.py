import cv2
import imutils
import mahotas
import numpy as np
from scipy.spatial import distance as dist


class Moments:
    def __init__(self, shape_file_dir):
        image = cv2.imread(shape_file_dir)
        _, self.shape_feature = self.describe_shape(image, init_feature=True)

    def describe_shape(self, image, visual=False, init_feature=False):
        shape_features = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)
        thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.erode(thresh, None, iterations=2)

        if visual:
            cv2.imshow("feature", thresh)
            cv2.waitKey(0)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if not init_feature:
            cnts = list(filter(is_prop_region, cnts))

        for c in cnts:
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            (x, y, w, h) = cv2.boundingRect(c)
            roi = mask[y:y + h, x:x + w]
            features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
            shape_features.append(features)
        shape_features = np.array(shape_features)
        return cnts, shape_features

    def find_loc(self, image, visual=False):
        cnts, features = self.describe_shape(image)
        if len(features) == 0:
            return 0, 0, 0, 0
        else:
            D = dist.cdist(self.shape_feature, features)
            i = np.argmin(D)
            c = cnts[i]
            (x, y, w, h) = cv2.boundingRect(c)
            # show found loc
            found_img = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 3)
            if visual:
                cv2.imshow("found", found_img)
                cv2.waitKey(0)
            return x + w // 2, y + h // 2, w, h

    @staticmethod
    def is_prop_region(c):
        if 150 < cv2.contourArea(c) < 500:
            return True
        else:
            return False












