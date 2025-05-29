import numpy as np
import torch

from model import MOSDC
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
import heapq
from utils import load_data


def train(args, device):
    print("dataset: " + args.dataset)
    repeat_num = args.n_repeated
    fea, labels, num_view, dimension = load_data(args.dataset, device)

    num_classes = len(np.unique(labels))
    sample_num = labels.shape[0]
    labels = labels.to(device)
    hid_d = [args.d1, args.d2, args.d3, args.d4, args.d5, args.d6]
    real_label_ratio = args.label_ratio
    if round(sample_num * real_label_ratio) < num_classes:
        real_label_ratio = real_label_ratio + args.select_each_ratio
    if round(sample_num * real_label_ratio) < num_classes:
        real_label_ratio = real_label_ratio + args.select_each_ratio
    if round(sample_num * real_label_ratio) < num_classes:
        real_label_ratio = real_label_ratio + args.select_each_ratio
    if round(sample_num * real_label_ratio) < num_classes:
        real_label_ratio = real_label_ratio + args.select_each_ratio
    sss = StratifiedShuffleSplit(n_splits=repeat_num, test_size=real_label_ratio, random_state=1)

    each_fea = []
    for v in range(num_view):
        each_fea.append(fea[v])

    com_acc1 = np.zeros((5, repeat_num))
    com_f1_1 = np.zeros((5, repeat_num))

    iter = -1
    repeat = 1
    for unlabel_index, label_index in sss.split(fea[0], labels.cpu().numpy()):
        print("Repeat: " + str(repeat))
        repeat = repeat +1
        iter = iter + 1
        real_unlabel_index = unlabel_index
        train_label = labels.cpu().numpy()
        train_label = torch.tensor(train_label)
        train_label = train_label.to(device)
        this_ratio = real_label_ratio
        model = MOSDC(num_classes, num_view, dimension, hid_d, device, args.k, args.dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        epoch_max = args.num_epoch
        top_ratio = args.top_ratio
        for cnttt in range(200):
            if this_ratio > 0.51:
                cnttt = cnttt - 1
                break
            loss_doc = []

            # ==================  train ========================
            with tqdm(total=epoch_max, desc="Training") as pbar:
                for epoch in range(epoch_max):
                    loss_total = 0
                    model.train()
                    optimizer.zero_grad()

                    specific_fea_de_lay2, view_class_specific_res, view_class_share_res, label_class_specific_res, label_class_share_res, specific_con, share_fea_en_lay2, edge_index_specific, edge_index_share = model(each_fea)
                    # encoder_decoder_loss
                    loss1 = nn.MSELoss()
                    each_fea1 = torch.cat(each_fea, dim=1)
                    specific_fea_de_lay2 = torch.cat(specific_fea_de_lay2, dim=1)
                    ed_loss = loss1(each_fea1, specific_fea_de_lay2)
                    del each_fea1, specific_fea_de_lay2

                    # classfy_view_loss
                    view_loss = 0
                    for v in range(num_view):
                        tmp_label = torch.zeros((sample_num, num_view), dtype=torch.float, device=device)
                        tmp_label[:, v] = 1.0
                        view_loss = view_loss + F.kl_div(view_class_specific_res[v].softmax(dim=-1).log(),
                                                         tmp_label.softmax(dim=-1), reduction='sum')
                    tmp_label = torch.ones((sample_num, num_view), dtype=torch.float, device=device) / num_view
                    view_loss = view_loss + F.kl_div(view_class_share_res.softmax(dim=-1).log(), tmp_label.softmax(dim=-1), reduction='sum')

                    # label_loss
                    label_loss = 0
                    label_loss = F.cross_entropy(label_class_specific_res[label_index], train_label[label_index].squeeze()) + F.cross_entropy(label_class_share_res[label_index], train_label[label_index].squeeze())
                    loss_total = ed_loss + args.lambda1 * view_loss + args.lambda2 * label_loss

                    # loss_tatal
                    loss_total.backward()
                    optimizer.step()
                    loss_doc.append(loss_total.item())

                    pbar.update(1)

            # ==================  test ========================
            with torch.no_grad():
                model.eval()
                acc = 0
                each_fea1 = []
                all_fea = []
                for v in range(num_view):
                    each_fea1.append(fea[v][real_unlabel_index])
                    all_fea.append(fea[v])
                _, _, _, label_class_specific_res, label_class_share_res, _, _, _, _ = model(each_fea1)
                class_res = torch.max(label_class_specific_res.softmax(dim=-1), label_class_share_res.softmax(dim=-1))

                pred = torch.argmax(class_res, 1).cpu().detach().numpy()

                ACC_test = accuracy_score(labels[real_unlabel_index].detach().cpu().numpy(), pred)
                F1_test = f1_score(labels[real_unlabel_index].detach().cpu().numpy(), pred, average='macro')

                if this_ratio >= 0.085 and this_ratio <= 0.115:
                    com_acc1[0, iter] = ACC_test
                    com_f1_1[0, iter] = F1_test

                if this_ratio >= 0.185 and this_ratio <= 0.215:
                    com_acc1[1, iter] = ACC_test
                    com_f1_1[1, iter] = F1_test

                if this_ratio >= 0.285 and this_ratio <= 0.315:
                    com_acc1[2, iter] = ACC_test
                    com_f1_1[2, iter] = F1_test

                if this_ratio >= 0.385 and this_ratio <= 0.415:
                    com_acc1[3, iter] = ACC_test
                    com_f1_1[3, iter] = F1_test

                if this_ratio >= 0.485 and this_ratio <= 0.515:
                    com_acc1[4, iter] = ACC_test
                    com_f1_1[4, iter] = F1_test

                print("Ratio: {:.2f}, ACC: {:.2f}, F1: {:.2f}".format(this_ratio, ACC_test * 100, F1_test * 100))
                add_pseudo = []
                class_sample = []
                for k in range(num_classes):
                    class_sample.append([])
                for num in range(label_index.shape[0]):
                    class_sample[labels[label_index[num]]].append(label_index[num])
                max_value1 = []
                max_value2 = []
                _, _, _, label_class_specific_res, label_class_share_res, specific_con, share_fea, _, _ = model(each_fea)
                label_class_specific_res = label_class_specific_res[unlabel_index].softmax(dim=-1)
                label_class_share_res = label_class_share_res[unlabel_index].softmax(dim=-1)

                cnt = 0
                all = 0
                tenporary = []
                for num in range(unlabel_index.shape[0]):
                    which1 = torch.argmax(label_class_specific_res[num])
                    which2 = torch.argmax(label_class_share_res[num])
                    if which1 == which2:
                        add_pseudo.append(unlabel_index[num])
                        tenporary.append(num)
                        max_value1.append(max(label_class_specific_res[num]))
                        max_value2.append(max(label_class_share_res[num]))
                top_number = min(round(sample_num * top_ratio), len(add_pseudo))
                value1 = heapq.nlargest(top_number, max_value1)
                value2 = heapq.nlargest(top_number, max_value2)

                psudo_add = []
                pre_l = []
                # top_ratio=top_ratio+0.01
                top_ratio = min(top_ratio, 0.2)
                for num in range(len(add_pseudo)):
                    if max_value1[num] in value1 and max_value2[num] in value2:
                        cnt = cnt + 1
                        psudo_add.append(add_pseudo[num])
                        pre_l.append(torch.argmax(label_class_specific_res[tenporary[num]]))
                        if torch.argmax(label_class_specific_res[tenporary[num]]) == labels[add_pseudo[num]]:
                            all = all + 1

                max_res = []
                for num in range(unlabel_index.shape[0]):
                    max_res.append(max(label_class_specific_res[num]) * max(label_class_share_res[num]))
                min_number = min(round(sample_num * args.select_each_ratio), unlabel_index.shape[0])
                min_value = heapq.nsmallest(min_number, max_res)
                select_sample = []
                if min_number >= 0:
                    for num in range(unlabel_index.shape[0]):
                        if max_res[num] in min_value:
                            select_sample.append(unlabel_index[num])
                    for num in range(len(select_sample)):
                        unlabel_index = np.delete(unlabel_index, np.where((unlabel_index == select_sample[num])))
                        real_unlabel_index = np.delete(real_unlabel_index, np.where((real_unlabel_index == select_sample[num])))
                        label_index = np.append(label_index, values=select_sample[num])

                if len(psudo_add) > 0 and this_ratio >= 0.175:
                    top_ratio = top_ratio + 0.0015
                    for num in range(len(psudo_add)):
                        train_label[psudo_add[num]] = pre_l[num]
                        unlabel_index = np.delete(unlabel_index, np.where((unlabel_index == psudo_add[num])))
                        label_index = np.append(label_index, values=psudo_add[num])
                this_ratio = this_ratio + args.select_each_ratio
                tmp_epoch = epoch_max - 15
                epoch_max = max(150, tmp_epoch)

    com_avg1 = np.mean(com_acc1, 1)
    com_std1 = np.std(com_acc1, 1)
    com_f1_avg1 = np.mean(com_f1_1, 1)
    com_f1_std1 = np.std(com_f1_1, 1)

    print("Final Result:")
    for i in range(5):
        print("Ratio: {:.2f}, ACC: {:.1f} ({:.1f}), F1: {:.1f} ({:.1f})".format((i+1)/10, com_avg1[i]*100, com_std1[i]*100, com_f1_avg1[i]*100, com_f1_std1[i]*100))










