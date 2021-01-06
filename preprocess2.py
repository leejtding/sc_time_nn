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

    train_len = len(train_cell_list[0])
    for tl_i in range(1,len(train_cell_list)):
        if len(train_cell_list[tl_i]) < train_len:
            train_len = len(train_cell_list[tl_i])

    test_len = len(test_cell_list[0])
    for tl_j in range(1,len(test_cell_list)):
        if len(test_cell_list[tl_j]) < test_len:
            test_len = len(test_cell_list[tl_j])

    train_input_list = []
    train_output_list = []
    test_input_list = []
    test_output_list = []

    for t_i in range(train_len):
        t_i_list = []
        for pt_i in range(len(train_cell_list)):
            t_i_list.append(train_cell_list[pt_i][t_i])
        t_i_input = t_i_list[:-1]
        t_i_output = t_i_list[1:]
        t_i_input_arr = np.vstack(t_i_input)
        t_i_output_arr = np.vstack(t_i_output)
        train_input_list.append(t_i_input_arr.astype(np.float32))
        train_output_list.append(t_i_output_arr.astype(np.float32))

    for t_j in range(test_len):
        t_j_list = []
        for pt_j in range(len(test_cell_list)):
            t_j_list.append(test_cell_list[pt_j][t_j])
        t_j_input = t_j_list[:-1]
        t_j_output = t_j_list[1:]
        t_j_input_arr = np.vstack(t_j_input)
        t_j_output_arr = np.vstack(t_j_output)
        test_input_list.append(t_j_input_arr.astype(np.float32))
        test_output_list.append(t_j_output_arr.astype(np.float32))

    train_input = np.array(train_input_list)
    train_output = np.array(train_output_list)
    test_input = np.array(test_input_list)
    test_output = np.array(test_output_list)

    return train_input, train_output, test_input, test_output

# ti, to, ti2, to2 = get_data('data/hsmm_data.csv', 'data/hsmm_times.csv')
# print(ti.shape)
# ti_tensor = torch.from_numpy(ti[0])
# print(ti_tensor.view(len(ti_tensor), 1, -1).shape)
# print(to.shape)
# print(ti2.shape)
# print(to2.shape)