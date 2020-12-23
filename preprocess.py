import random
import numpy as np
import pandas as pd
import torch

def get_data(data_path, times_path, seed=123):

    expressions_df = pd.read_csv(data_path).T
    expressions = expressions_df.values[1:]
    times = pd.read_csv(times_path).values

    time_dict = {}
    for i in range(len(times)):
        if times[i][1] not in time_dict.keys():
            time_dict[times[i][1]] = [i]
        else:
            time_dict[times[i][1]].append(i)

    random.seed(seed)
    train_cell_list = []
    train_label_list = []
    test_cell_list = []
    test_label_list = []
    test_num = int((len(times) * 0.2) // len(time_dict.keys()))

    for pt in time_dict:
        pt_inds = time_dict[pt]
        pt_inds2 = list(range(len(pt_inds)))

        sample_indices = random.sample(pt_inds2, test_num)
        pt_inds_arr = np.array(pt_inds)
        test_pt_indices = pt_inds_arr[sample_indices]
        train_pt_indices = np.delete(pt_inds_arr, sample_indices, 0)

        train_pts = expressions[train_pt_indices]
        train_cell_list.append(train_pts)
        train_label_list.append(times[train_pt_indices][:,1])

        test_pts = expressions[test_pt_indices]
        test_cell_list.append(test_pts)
        test_label_list.append(times[test_pt_indices][:,1])

    train_cells_arr = np.vstack(train_cell_list)
    train_labels_arr = np.concatenate(train_label_list)
    test_cells_arr = np.vstack(test_cell_list)
    test_labels_arr = np.concatenate(test_label_list)

    time_point_list = list(time_dict.keys())
    for train_i in range(len(train_labels_arr)):
        train_labels_arr[train_i] = time_point_list.index(train_labels_arr[train_i])
    for test_i in range(len(test_labels_arr)):
        test_labels_arr[test_i] = time_point_list.index(test_labels_arr[test_i])

    train_cells = torch.from_numpy(train_cells_arr.astype(np.float32))
    train_labels = torch.from_numpy(train_labels_arr.astype(np.float32))
    test_cells = torch.from_numpy(test_cells_arr.astype(np.float32))
    test_labels = torch.from_numpy(test_labels_arr.astype(np.float32))

    return train_cells, train_labels, test_cells, test_labels

# tc, tl, tc2, tl2 = get_data('data/hsmm_data.csv', 'data/hsmm_times.csv')
# print(tc.shape)
# print(tl.shape)
# print(tc2.shape)
# print(tl2.shape)