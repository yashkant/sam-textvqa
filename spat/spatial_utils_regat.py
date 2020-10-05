import math

import numpy as np
import torch
from torch.autograd import Variable


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


def build_graph_using_normalized_boxes(bbox, label_num=11, distance_threshold=0.5):
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
    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    # [num_boxes, 1]
    # bbox_width = xmax - xmin
    # bbox_height = ymax - ymin
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
        adj_matrix[i, i] = 12
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
                adj_matrix[i, j] = 1  # covers
                adj_matrix[j, i] = 2  # inside
            # class 2: cover (j covers i)
            elif (
                xmin[j] < xmin[i]
                and xmax[j] > xmax[i]
                and ymin[j] < ymin[i]
                and ymax[j] > ymax[i]
            ):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)

                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
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
                        # if (adj_matrix[i, j] > 0):
                        adj_matrix[i, j] = int(np.ceil(label_i / (math.pi / 4))) + 3
                        adj_matrix[j, i] = int(np.ceil(label_j / (math.pi / 4))) + 3
    return adj_matrix


def torch_broadcast_adj_matrix(adj_matrix, label_num=11, device=torch.device("cuda")):
    """broudcast spatial relation graph

    # (YK): Changed to work on single-instance as well

    Args:
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:
        result: [batch_size,num_boxes, num_boxes, label_num]


    # (YK): Build one-hot from classes [1-12] to [0-11]

    """
    result = []
    for i in range(1, label_num + 1):
        index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze()
        curr_result = torch.zeros_like(adj_matrix)
        curr_result = curr_result.view(-1)
        curr_result[index] += 1
        result.append(curr_result.view(adj_matrix.shape + (1,)))
    result = torch.cat(result, dim=-1)
    return result


def _build_replace_dict():
    share_replace_dict = {
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


def build_graph_using_normalized_boxes_new(
    bbox, label_num=11, distance_threshold=0.5, build_shared=[1, 3, 5, 7, 9]
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
    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    share_replace_dict = _build_replace_dict()
    adj_matrix_shared = {}

    for key in share_replace_dict.keys():
        adj_matrix_shared[key] = np.zeros((num_box, num_box))

    # adj_matrix_share3_1 = np.zeros((num_box, num_box))
    # adj_matrix_share3_2 = np.zeros((num_box, num_box))

    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    # [num_boxes, 1]
    # bbox_width = xmax - xmin
    # bbox_height = ymax - ymin
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
        adj_matrix[i, i] = 12
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
                adj_matrix[i, j] = 1  # covers
                adj_matrix[j, i] = 2  # inside
            # class 2: cover (j covers i)
            elif (
                xmin[j] < xmin[i]
                and xmax[j] > xmax[i]
                and ymin[j] < ymin[i]
                and ymax[j] > ymax[i]
            ):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)

                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
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
                        # if (adj_matrix[i, j] > 0):
                        adj_matrix[i, j] = int(np.ceil(label_i / (math.pi / 4))) + 3
                        adj_matrix[j, i] = int(np.ceil(label_j / (math.pi / 4))) + 3

                        # fill in share spatial-matrices
                        for key in adj_matrix_shared.keys():
                            adj_matrix_shared[key][i, j] = share_replace_dict[key].get(
                                adj_matrix[i, j], 0
                            )
                            adj_matrix_shared[key][j, i] = share_replace_dict[key].get(
                                adj_matrix[j, i], 0
                            )

    for key in adj_matrix_shared.keys():
        adj_matrix_shared[key] = adj_matrix_shared[key].astype(np.int8)

    return adj_matrix.astype(np.int8), adj_matrix_shared


def random_spatial_processor(pad_obj_ocr_bboxes):
    randomize = [1, 3]
    adj_matrix_random1_shape = (len(pad_obj_ocr_bboxes), len(pad_obj_ocr_bboxes), 1)
    adj_matrix_random3_shape = (len(pad_obj_ocr_bboxes), len(pad_obj_ocr_bboxes), 3)
    adj_matrix_random1 = np.zeros(adj_matrix_random1_shape, dtype=np.int8)
    adj_matrix_random3 = np.zeros(adj_matrix_random3_shape, dtype=np.int8)
    spatial_relations_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for matrix, _randomize in zip([adj_matrix_random1, adj_matrix_random3], randomize):
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                random_indices = np.random.choice(
                    spatial_relations_types, size=_randomize, replace=False
                )
                # remove none-edges
                if 0 not in random_indices:
                    matrix[row][col] = random_indices

    # Remove masked relations
    masked_inds = np.where(pad_obj_ocr_bboxes.sum(axis=-1) == 0)
    adj_matrix_random1[masked_inds] = 0
    adj_matrix_random1[:, masked_inds] = 0
    adj_matrix_random3[masked_inds] = 0
    adj_matrix_random3[:, masked_inds] = 0
    return adj_matrix_random1.astype(np.int8), adj_matrix_random3.astype(np.int8)


def torch_extract_position_embedding(
    position_mat, feat_dim, wave_length=1000, device=torch.device("cuda")
):
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,)) * wave_length, (8.0 / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)
    div_mat = torch.div(position_mat.to(device), dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(
        embedding.shape[0], embedding.shape[1], embedding.shape[2], feat_dim
    )
    return embedding


def torch_extract_position_matrix(bbox, nongt_dim=36):
    """Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 4]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1)
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [batch_size,num_boxes, num_boxes]
    delta_x = center_x - torch.transpose(center_x, 1, 2)
    delta_x = torch.div(delta_x, bbox_width)

    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)
    delta_y = center_y - torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)
    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)
    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


def prepare_graph_variables(
    relation_type,
    bb,
    sem_adj_matrix,
    spa_adj_matrix,
    num_objects,
    nongt_dim,
    pos_emb_dim,
    spa_label_num,
    sem_label_num,
    device,
):
    pos_emb_var, sem_adj_matrix_var, spa_adj_matrix_var = None, None, None
    if relation_type == "spatial":
        assert spa_adj_matrix.dim() > 2, "Found spa_adj_matrix of wrong shape"
        spa_adj_matrix = spa_adj_matrix.to(device)
        spa_adj_matrix = spa_adj_matrix[:, :num_objects, :num_objects]
        spa_adj_matrix = torch_broadcast_adj_matrix(
            spa_adj_matrix, label_num=spa_label_num, device=device
        )
        spa_adj_matrix_var = Variable(spa_adj_matrix).to(device)
    if relation_type == "semantic":
        assert sem_adj_matrix.dim() > 2, "Found sem_adj_matrix of wrong shape"
        sem_adj_matrix = sem_adj_matrix.to(device)
        sem_adj_matrix = sem_adj_matrix[:, :num_objects, :num_objects]
        sem_adj_matrix = torch_broadcast_adj_matrix(
            sem_adj_matrix, label_num=sem_label_num, device=device
        )
        sem_adj_matrix_var = Variable(sem_adj_matrix).to(device)
    else:
        # (YK): Todo what's this here?
        bb = bb.to(device)
        pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim)
        pos_emb = torch_extract_position_embedding(
            pos_mat, feat_dim=pos_emb_dim, device=device
        )
        pos_emb_var = Variable(pos_emb).to(device)
    return pos_emb_var, sem_adj_matrix_var, spa_adj_matrix_var
