import numpy
import peakutils


def peak_index(in_datas):
    indexes = peakutils.indexes(in_datas, thres=0.2, min_dist=10)
    return indexes

