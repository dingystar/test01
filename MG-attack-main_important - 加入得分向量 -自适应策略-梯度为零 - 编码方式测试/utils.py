import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
import data.io as io
import copy
import arguments
import math

args = arguments.parse_args()


def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx


def exclude_idx(idx: np.ndarray, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
            idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
        exclude_idx(idx, [train_idx]),
        nstopping, replace=False)
    return train_idx, stopping_idx


def gen_splits(labels: np.ndarray, idx_split_args,
               test: bool = False):
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
        all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
        known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx


def load_data(graph_name='cora_ml', lbl_noise=0, str_noise_rate=0.1, seed=2144199730):#传递过来的参数依靠原定义中的参数
    print("graph_name,lbl_noise,str_noise,seed=", graph_name, lbl_noise, str_noise_rate, seed)
    dataset = io.load_dataset(graph_name)   #加载已存在数据集
    print("dataset=",dataset)
    dataset.standardize(select_lcc=True)    #
    features = dataset.attr_matrix   #节点的特征矩阵
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    # features = sparse_mx_to_torch_sparse_tensor(features)
    labels = dataset.labels
    adj = dataset.adj_matrix   #邻接矩阵
    print(features)
    # print(adj)
    # os.system("pause")
    adj_matrix = copy.deepcopy(adj.toarray())
    num_edges = np.sum(adj_matrix) // 2
    print(num_edges,adj_matrix[0])
    # os.system("pause")
    #print("adj_matix=",adj_matrix)
    #os.system("pause")
    adj = str_noise(adj, labels, str_noise_rate, seed)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if graph_name == 'ms_academic':
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 5000, 'seed': seed}
    else:
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': seed}
    #print("idx_split_args=",idx_split_args)
    idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)
    print("len(train)=",len(idx_train),"len(val)=",len(idx_val),"len(test)=",len(idx_test))
    # os.system("pause")
    n_class = max(labels) + 1

    # labels = add_label_noise(idx_train, labels, lbl_noise, seed)
    print("test1")
    labels,res_eva = gradient_max(adj, features, idx_train, labels, lbl_noise, adj_matrix)  # adj, idx_train, labels, noise_num对标签添加扰动信息 扰动数量节点标签设置为一个不存在值，选择方式，依据最大梯度

    # print("扰动节点数为",len(res_eva),"训练集数",len(idx_train))
    # os.system("pause")

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # print("adj=",adj)
    # print("features=",features)
    # print("labels=",labels)
    # print("idx_train=",idx_train)
    # print("idx_val=",idx_val)
    # print("idx_test=",idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test   #=邻接矩阵。特征矩阵，标签，训练集，验证集，测试集


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    c = correct
    correct = correct.sum()
    acc = correct / len(labels)
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def similarity_matrix(features, gamma):
    tmp = features @ features.T
    n_data = features.shape[0]
    diag = np.diag(tmp)
    S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
    return np.exp(S)

def diagnoal(similarity_matrix):
    D = np.diag(np.sum(similarity_matrix, axis=1))
    return D

def func_gradient(adj, features, labels, idx_train, y_l):
    # calculate the objective function value
    y_0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx_train:
        y_0[i][labels[i]] = 1.0
    y_lo = y_0.float()

    idx_all = np.arange(labels.shape[0])
    idx_other = np.delete(idx_all, idx_train)
    # print("labels=",labels,labels.shape[0])
    # print("idx_train=",idx_train,len(idx_train))
    # print("idx_all=",idx_all,len(idx_all))
    # print("idx_other=",idx_other[91],len(idx_other))
    # os.system("pause")
    #idx_other_t = torch.from_numpy(idx_other)

    # ground truth
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for j in idx_other:
        y0[j][labels[j]] = 1.0
    y_uo = y0.float()
    #y_u_u = torch.index_select(y_uo, 0, idx_other_t)
    y_u_u = y_uo[idx_other]


    #Similarity Metrix
    features_np = features.numpy()
    # print("features_np=",features_np,features_np.shape,features_np[0][0])
    # os.system("pause")
    S = similarity_matrix(features_np, gamma=0.1)
    # S=S.numpy()
    D = diagnoal(S)
    y_tr = labels[idx_train]
    # print("y_tr=",y_tr,len(y_tr))
    # print("D=",D)
    # os.system("pause")
    n_tr = len(y_tr)
    Suu = S[n_tr:, n_tr:]
    Duu = D[n_tr:, n_tr:]
    Sul = S[n_tr:, :n_tr]
    SM_A = np.linalg.inv(Duu - Suu) @ Sul
    SM_A = torch.from_numpy(SM_A).float()
    y_pre_sm = SM_A @ y_lo[idx_train] #original predicted y_u
    y_sm_u = SM_A @ y_l[idx_train] #A@adv_y

    # y_sm = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    # y_sm[idx_other] = y_sm_u


    #APPNP
    for _ in range(args.K):
        y_appo = torch.matmul(adj, y_lo)
        y_appo = (1 - args.alpha) * y_appo + args.alpha * y_lo
    #y_pre_app = torch.index_select(y_appo, 0, idx_other_t) #original predicted y_u
    y_pre_app = y_appo[idx_other]

    for _ in range(args.K):
        advy_app = torch.matmul(adj, y_l)
        advy_app = (1 - args.alpha) * advy_app + args.alpha * y_l
    #y_app_u = torch.index_select(advy_app, 0, idx_other_t) #A@adv_y
    y_app_u = advy_app[idx_other]


    #S^K
    SK_A = adj.pow(10)
    y_sk_o = SK_A @ y_lo
    #y_pre_sk = torch.index_select(y_sk_o, 0, idx_other_t) #original predicted y_u
    y_pre_sk = y_sk_o[idx_other]

    y_sk = SK_A @ y_l
    #y_sk_u = torch.index_select(y_sk, 0, idx_other_t) #A@adv_y
    y_sk_u = y_sk[idx_other]


    # SM - SP
    #tmp = y_sm_u - y_pre_app
    #SK - SP
    #tmp = y_sk_u - y_pre_app


    #SM - SM
    #tmp = y_sm_u - y_pre_sm
    #SK - SM
    #tmp = y_sk_u - y_pre_sm
    # SP - SM
    #tmp = y_app_u - y_pre_sm

    #SM-SK
    # tmp = y_sm_u - y_pre_sk   #只有pmbed用这个
    #SK-SK
    # tmp = y_sk_u - y_pre_sk
    #SP-SK
    # tmp = y_app_u - y_pre_sk


    #SM - yu
    tmp = y_sm_u - y_u_u   #其他用这个
    # SK - yu
    # tmp = y_sk_u - y_u_u
    # SP - yu
    # tmp = y_app_u - y_u_u

    f = -0.5 * torch.sum(tmp * tmp)
    return f



def gradient_max(adj, features, idx_train, labels, noise_num, adj_matrix):
    #print("梯度计算")
    str_set = []
    if noise_num == 0:  # fix a corner case
        return labels,0
    nclass = max(labels) + 1
    labels_new = labels.copy()
    RWRRES = RWR(adj, features, idx_train, labels,noise_num, adj_matrix)
    # print("RWRRES=",RWRRES)
    for _ in range(2):
        flag_g0 = 0
        # print("labels=",labels_new.shape[0], labels_new.max().item() + 1)
        y0 = torch.zeros(size=(labels_new.shape[0], labels_new.max().item() + 1))
        # print("y0=",y0[0])
        # print(idx_train)
        for i in idx_train:
            y0[i][labels_new[i]] = 1.0
        y_l = y0.float()
        # print("y_l=",y_l[91])
        # print("adj=",adj.shape)
        # print("features=",features.shape)
        # os.system("pause")
        #print("y_l=", y_l,len(y_l[22]),y_l[22])
        adv_y = Variable(y_l, requires_grad=True)
        #print("adv_y=",adv_y)
        f = func_gradient(adj, features, labels, idx_train, adv_y)
        #print("f=",f,adv_y)
        grad = torch.autograd.grad(f, adv_y, retain_graph=False, create_graph=False)[0]
        # print("grad=",grad,grad.shape)
        grad_train = []
        for i in idx_train:
            # grad_train.append(grad[i][labels_new[i]].tolist())  # list
            grad_train.append([i,grad[i][labels_new[i]].tolist()])  # list
        print("梯度值=",grad_train,len(grad_train))
        evalution = copy.deepcopy(grad_train)
        evalution = sorted(evalution,key=lambda x: x[1])    #尝试处理当梯度存在负值又存在正值

        print("排序梯度",evalution)

        neg_max = float('-inf');pos_max = float('-inf');max_abs = float('-inf')
        for i in evalution:
            if abs(i[-1]) > max_abs:
                max_abs = abs(i[-1])
        if  evalution[-1][-1] - evalution[0][-1] >50 or max_abs > 50:#30   1.改为50  为负时直接除以neg_max   2 . 每个数减去最小值除以区间长度
            flag_g0 = 1
        tmp_min = evalution[0][-1]
        print("最小值测试",tmp_min)
        if flag_g0 == 1:
            for i in evalution:
                if(i[-1] < 0) and neg_max <i[-1]:
                    neg_max = i[-1]
                elif i[-1] > 0 and pos_max < i[-1]:
                    pos_max = i[-1]
            print("flag0=",flag_g0)
            print("neg_max=",neg_max,"pos_max=",pos_max)
            for x in evalution :
                if x[-1] < 0:
                    x[-1] = (1/x[-1])/(1/neg_max)
                else:
                    x[-1] = (x[-1] / pos_max)  #*0.05
            print("pumbed梯度转换=",evalution)

            # for x in evalution:
            #     if x[-1] < neg_max:
            #         neg_max = x[-1]
            #     elif x[-1] > pos_max:
            #         pos_max = x[-1]
            # for i in evalution:
            #     i[-1] = (i[-1] - neg_max) / (pos_max - neg_max)
            # print("neg_max=",neg_max,"pos_max=",pos_max)
            # print("pumbed梯度转换=",evalution)
        else:
            for x in range(len(evalution)):
                evalution[x][1] = evalution[x][1] - tmp_min
                # if evalution[x][1] == 0:
                #     evalution[x][1] = evalution[x][1]
                # else:
                #     evalution[x][1] = 1 / abs(evalution[x][1])
        #         print("evalution[x][1]=",evalution[x][1],evalution[x])
        # print("梯度",evalution)
        #evalution = [[x[0], 1 / abs(x[1])] for x in evalution]
        evalution = sorted(evalution,key=lambda x: x[1])
        max_value = evalution[-1][-1]
        # print("max_value",max_value)
        # print("2次",evalution)
        # evalution = [[x[0], x[1] / max_value] for x in evalution]
        for i in range(len(evalution)):
            if (evalution[i][1] != 0):
                # print("测试",evalution[i][1])
                # print("最值",max_value)
                evalution[i][1] = evalution[i][1] / max_value   #1
            else:
                # print("测试",evalution[i][1])
                # print("最值",max_value)
                evalution[i][1] = 0   #1
        # print("梯度转换",evalution)
        evalution = sorted(evalution,key=lambda x: x[1])
        print("梯度最终排序",evalution)
        # os.system("pause")
        RWRRES = RWRRES.squeeze()
        RW_score = []
        for i in idx_train:
            RW_score.append([i,RWRRES[i].item()])
        RW_score = sorted(RW_score, key=lambda x: x[1])
        print("score=", RW_score,len(RW_score),len(idx_train))
        # os.system("pause")
        max_value2 = RW_score[-1][-1]
        RW_score = [[x[0], x[1] / max_value2] for x in RW_score]
        res_eva = []
        for i in evalution:
            for j in RW_score:
                if i[0] == j[0]:
                    res_eva.append([i[0], 0.9 * i[-1] + 0.1 * j[-1]])
                    break
        temp_num = 0
        if noise_num > 5:
            temp_num = int((noise_num - 5)/2 * nclass)
        else:
            temp_num = noise_num * nclass
        print("最终结果",res_eva)
        res_eva = sorted(res_eva,key=lambda x: x[1])[-temp_num:]
        for i in res_eva:
            labels_new[i[0]] = max(labels)
            str_set.append(i)
    return labels_new,str_set # narray


def str_noise(adj, labels, noise_rate, seed=0):
    if noise_rate > 1.0:
        return adj
    idx = np.arange(len(labels))
    adj = adj.tocoo().astype(np.float32)

    row = adj.row
    col = adj.col

    upper_edge = np.arange(len(row))[row < col]
    idx_upper_edge = np.arange(len(upper_edge))
    good_edge_idx = idx_upper_edge[labels[row[upper_edge]] == labels[col[upper_edge]]]
    bad_edge_idx = idx_upper_edge[labels[row[upper_edge]] != labels[col[upper_edge]]]

    origin_noise_rate = len(bad_edge_idx) / len(idx_upper_edge)

    rnd_state = np.random.RandomState(seed)
    random.seed(seed)
    if origin_noise_rate > noise_rate:
        sub_num = int(len(upper_edge) * (origin_noise_rate - noise_rate))
        inv_idx = rnd_state.choice(bad_edge_idx, sub_num, replace=False)
        for i in inv_idx:
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5:
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if col_new[0] != row_i:
                    col_new = col_new[0]
                else:
                    col_new = col_new[1]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        row[j] = col_new
                        break
            else:
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if row_new[0] != col_i:
                    row_new = row_new[0]
                else:
                    row_new = row_new[1]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        col[j] = row_new
                        break
    else:
        add_num = int(len(upper_edge) * (noise_rate - origin_noise_rate))
        inv_idx = rnd_state.choice(good_edge_idx, add_num, replace=False)
        for i in inv_idx:
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5:
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                col_new = col_new[0]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        row[j] = col_new
                        break
            else:
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                row_new = row_new[0]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        col[j] = row_new
                        break

    adj.row = row
    adj.col = col
    return adj


def get_noise_rate(adj, labels):
    indices = adj._indices().numpy()
    upper = indices[0, :] > indices[1, :]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    return bad_num / upper_indices.shape[1]