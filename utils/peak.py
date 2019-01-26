import numpy as np
import peakutils


def peak_index(in_datas):
    indexes = peakutils.indexes(np.asarray(in_datas, dtype=np.float32), thres=0.2, min_dist=10)
    return indexes

