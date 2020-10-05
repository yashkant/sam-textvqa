import math

import numpy as np
import torch


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    assert (boxBArea + boxAArea - interArea) != 0

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def torch_broadcast_adj_matrix(adj_matrix):
    """broudcast spatial relation graph

    # (YK): Changed to work on single-instance as well

    Args:
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:
        result: [batch_size,num_boxes, num_boxes, label_num]


    # (YK): Build one-hot from classes [1-12] to [0-11]

    """
    result = torch.zeros_like(adj_matrix).unsqueeze(-1).repeat(1, 1, 12)
    _indices = (adj_matrix - 1) * (adj_matrix > 0)
    result.scatter_(2, _indices.long().unsqueeze(-1), 1)
    result[:, :, 0] = result[:, :, 0] * (adj_matrix > 0)
    return result


def _build_replace_dict():
    share_replace_dict = {
        "1": {},
        "31": {},
        "32": {},
        "51": {},
        "52": {},
        "71": {},
        "72": {},
        "91": {},
        "92": {},
    }

    for quad in [4, 5, 6, 7, 8, 9, 10, 11]:
        share_replace_dict["31"][quad] = quad + 1
        share_replace_dict["32"][quad] = quad - 1

        share_replace_dict["51"][quad] = quad + 2
        share_replace_dict["52"][quad] = quad - 2

        share_replace_dict["71"][quad] = quad + 3
        share_replace_dict["72"][quad] = quad - 3

        share_replace_dict["91"][quad] = quad + 4
        share_replace_dict["92"][quad] = quad - 4

    adjust_sectors = {0: 8, 1: 9, 2: 10, 3: 11, 12: 4, 13: 5, 14: 6, 15: 7}

    for _, value in share_replace_dict.items():
        for key, val in value.items():
            if val < 4 or val > 11:
                assert val in adjust_sectors
                value[key] = adjust_sectors[val]

    return share_replace_dict


def build_graph_using_normalized_boxes(
    bbox, label_num=11, distance_threshold=0.5, build_gauss_bias=False
):
    """Build spatial graph
    Args:
        bbox: [num_boxes, 4]
    Returns:
        adj_matrix: [num_boxes, num_boxes, label_num]

    # (YK): Fixed by @harsh
    Remember
        - Adjacency matrix [j, i] means relationship of j (target/blue) with respect to i (origin/red) (j is W-NW of i and so on)
        - j_box is blue
        - i_box is red
        - blue arrow is from i_box (red box) to j_box (blue box) i.e from  origin to target
        - red arrow is from j_box (blue box) to i_box (red box) i.e from target to origin
    """
    # map mean for each sector
    mean_map = {}
    for sector in range(4, 12):
        mean_map[sector] = (math.pi / 8.0) * (2 * (sector - 4) + 1)
    num_box = bbox.shape[0]

    # adj_matrix = np.zeros((num_box, num_box))
    share_replace_dict = _build_replace_dict()
    adj_matrix_shared = {}
    # gauss_bias_shared = {}

    for key in share_replace_dict.keys():
        adj_matrix_shared[key] = np.zeros((num_box, num_box))

    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)

    # normalized coordinates
    image_h = 1.0
    image_w = 1.0
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(image_h ** 2 + image_w ** 2)
    for i in range(num_box):
        bbA = bbox[i]
        # (YK): Padded bbox
        if sum(bbA) == 0:
            continue
        adj_matrix_shared["1"][i, i] = 12
        for j in range(i + 1, num_box):
            bbB = bbox[j]
            # (YK): Padded bbox
            if sum(bbB) == 0:
                continue
            # class 1: inside (j inside i)
            if (
                xmin[i] < xmin[j]
                and xmax[i] > xmax[j]
                and ymin[i] < ymin[j]
                and ymax[i] > ymax[j]
            ):
                adj_matrix_shared["1"][i, j] = 1  # covers
                adj_matrix_shared["1"][j, i] = 2  # inside
            # class 2: cover (j covers i)
            elif (
                xmin[j] < xmin[i]
                and xmax[j] > xmax[i]
                and ymin[j] < ymin[i]
                and ymax[j] > ymax[i]
            ):
                adj_matrix_shared["1"][i, j] = 2
                adj_matrix_shared["1"][j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)

                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix_shared["1"][i, j] = 3
                    adj_matrix_shared["1"][j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff) ** 2 + (x_diff) ** 2)
                    if diag < distance_threshold * image_diag:
                        sin_ij = y_diff / diag
                        cos_ij = x_diff / diag
                        # first quadrant
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = math.pi + label_i
                        # fourth quadrant
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        # second quadrant
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = label_i + math.pi
                        # third quadrant
                        else:
                            label_i = 2 * math.pi - np.arccos(cos_ij)
                            label_j = label_i - math.pi
                        # goes from [1-8] + 3 -> [4-11]
                        # if (adj_matrix_shared["1"][i, j] > 0):
                        adj_matrix_shared["1"][i, j] = (
                            int(np.ceil(label_i / (math.pi / 4))) + 3
                        )
                        adj_matrix_shared["1"][j, i] = (
                            int(np.ceil(label_j / (math.pi / 4))) + 3
                        )

                        # fill in share spatial-matrices
                        for key in adj_matrix_shared.keys():
                            if key != "1":
                                adj_matrix_shared[key][i, j] = share_replace_dict[
                                    key
                                ].get(adj_matrix_shared["1"][i, j], 0)
                                adj_matrix_shared[key][j, i] = share_replace_dict[
                                    key
                                ].get(adj_matrix_shared["1"][j, i], 0)

    for key in adj_matrix_shared.keys():
        adj_matrix_shared[key] = adj_matrix_shared[key].astype(np.int8)

    return adj_matrix_shared
