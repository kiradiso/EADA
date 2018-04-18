import os
import math
import model
import data_loader
import torch
import torch.nn as nn
# import logger as Log
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.autograd import Variable


torch.manual_seed(1)
torch.cuda.manual_seed(1)
# parameters
"""
 For the Office-31 dataset, the S/T name should like : 'amazon/images', image_path is "Original_images"
 For the Office-Caltech dataset, the S/T name should like : 'amazon', image_path is "office_caltech_10"
 the datasets can be downloaded in the http as readme.txt showed.
"""
S_name = 'webcam'
T_name = 'caltech'
# dataset_path = "./office31_decaf7"
image_path = "office_caltech_10"

use_gpu = True
full_test = True
use_gd = True
use_gyd = True
use_dec = False
use_feature = False  # False means use images as input
use_ls = False  # use least square loss
# use_gls = False # use generalized loss sensitive loss
# use_w = False # use wasserstein distance loss
# slope_lrelu = 1e-2
# tmp = 3
# lambda_gp = 10 # gradient penalty for w/loss sensitive GAN
use_tensorboard = False
tb_path = "./log_base"
ndr_path = "./acc/max.txt"
train_test_split = False # whether split train and test set
batch_size = 64
image_size = 224          # if use images as input
inp_feature_size = 4096  # if use features as input , show the features num
c_size = [512]
gd_size = [1024, 1024]
feature_size = 256
class_num = 31   # close set problem
train_test_pp = 10
n_c = 3
lr_g = 0.01
lr_d = 0.01
lr_c = 0.01
lr_dec = 0.01
lambda_dec = 0.5
lambda_cnf = 0.5
epoch = 6000
show_step = 100
test_step = 200

ls = os.listdir(os.path.join(image_path, S_name))


semantic_consistency_loss = 'L1'
gls_dis_loss = 'L1'
gp_type = "L2" # min or L2


# function
def get_label_dict(ls):
    dc = {}
    for i in range(len(ls)):
        dc[ls[i]] = i
    return dc

def entropy_witht(x, t):
    px = F.softmax(x/t, dim=1)
    return -(px*torch.log(px)).sum(dim=1)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.1)


def mse_withw(x, weight, bound=1):
    """for least square GAN"""
    return torch.mean(weight*torch.pow(x-bound, 2))


def v2np(x):
    return x.data.cpu().numpy()


def get_norm_gradient_penalty(net, x, gamma, cuda):
    if cuda:
        x = x.cuda()
    x = Variable(x, requires_grad=True)

    output = net(x)
    gradOutput = torch.ones(output.size()).cuda() if cuda else torch.ones(output.size())
    gradient = \
    torch.autograd.grad(outputs=output, inputs=x, grad_outputs=gradOutput, create_graph=True, retain_graph=True,
                        only_inputs=True)[0]

    gradientPenalty = (gradient-1).norm(2, dim=1).mean() * gamma

    return gradientPenalty


def get_direct_gradient_penalty(net, x, gamma, cuda):
    if cuda:
        x = x.cuda()
    x = Variable(x, requires_grad=True)

    output = net(x)
    gradOutput = torch.ones(output.size()).cuda() if cuda else torch.ones(output.size())
    gradient = \
    torch.autograd.grad(outputs=output, inputs=x, grad_outputs=gradOutput, create_graph=True, retain_graph=True,
                        only_inputs=True)[0]

    gradientPenalty = gradient.norm(2, dim=1).mean() * gamma

    return gradientPenalty

# train data
if not use_feature:
    if train_test_split:
        pass
        # s_train_dtl, s_test_dtl, _ = data_loader.get_loader(
        #     batch_size, image_size, pp=train_test_pp, dataset_name=S_name, full_test=full_test)
        # t_train_dtl, t_test_dtl, _ = data_loader.get_loader(
        #     batch_size, image_size, pp=train_test_pp, dataset_name=T_name, full_test=full_test)
    else:
        flag_arr = np.random.randint(0, train_test_pp, 5000)
        labels = get_label_dict(ls)
        s_train_dtl = data_loader.get_fullloader(image_size, batch_size, image_path=image_path, dataset_name=S_name,
                                                  labels_dict=labels)
        s_test_dtl = data_loader.get_loader_witharr(batch_size, flag_arr == 0, image_size, False,
                                                    image_path, S_name, labels=labels,
                                                    dl=False)  # full label test
        t_train_dtl = data_loader.get_fullloader(image_size, batch_size, train=True, image_path=image_path,
                                                 dataset_name=T_name, labels_dict=labels)
        t_test_dtl = data_loader.get_fullloader(image_size, batch_size, train=False, image_path=image_path,
                                                dataset_name=T_name, labels_dict=labels, shuffle=False, dl=False)
else:
    pass
    # s_train_dtl = data_loader.get_featureloader(batch_size=batch_size, dataset_name=S_name, dataset_path=dataset_path)
    # s_test_dtl = data_loader.get_featureloader(dataset_name=S_name, shuffle=False, dataset_path=dataset_path)
    # t_train_dtl = data_loader.get_featureloader(batch_size=batch_size, dataset_name=T_name, dataset_path=dataset_path)
    # t_test_dtl = data_loader.get_featureloader(dataset_name=T_name, shuffle=False, dataset_path=dataset_path)

# loss
sc_loss = nn.L1Loss() if semantic_consistency_loss == 'L1' else nn.MSELoss()
clf_loss = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
domain_clf_loss = nn.BCELoss() if not use_ls else nn.MSELoss()

# input
source_label = Variable(torch.FloatTensor(batch_size))
target_label = Variable(torch.FloatTensor(batch_size))
onehot_label = Variable(torch.FloatTensor(batch_size, class_num))
source_label.data.fill_(1)
target_label.data.fill_(0)

# network
to_prob = True
G_f = model.Generator_Alx(feature_size=feature_size, fine_tune=True)
G_d = model.Classifier_withsize(feature_size=feature_size, size=gd_size, to_prob=to_prob)
C = model.Classifier_withsize(feature_size=feature_size, size=c_size, class_num=class_num, to_prob=False)
if use_gyd:
    G_yd = model.Classifier_withsize(feature_size=feature_size+class_num, size=gd_size, to_prob=to_prob)
if use_dec:
    Dec = model.Decoder(feature_size=feature_size, ini_imsize=image_size//4)

if use_gpu:
    sc_loss = sc_loss.cuda()
    clf_loss = clf_loss.cuda()
    G_f.cuda()
    G_d.cuda()
    C.cuda()
    softmax.cuda()
    if use_gyd:
        G_yd.cuda()
    if use_dec:
        Dec.cuda()
    source_label = source_label.cuda()
    target_label = target_label.cuda()
    domain_clf_loss = domain_clf_loss.cuda()

if use_feature:
    G_f_optim = torch.optim.SGD(G_f.parameters(), lr_g, momentum=0.9, weight_decay=1e-4)
    G_f.apply(weight_init)
else:
    G_f_optim = torch.optim.SGD([
        {'params': G_f.features.parameters(), 'lr': lr_g/10},
        {'params': G_f.linear.parameters(), 'lr': lr_g/10},
        {'params': G_f.extra.parameters()}
    ], lr=lr_g, momentum=0.9, weight_decay=1e-4)
    G_f.extra.apply(weight_init)
G_d_optim = torch.optim.SGD(G_d.parameters(), lr_d, momentum=0.9, weight_decay=1e-4)
G_d.apply(weight_init)
C_optim = torch.optim.SGD(C.parameters(), lr_c, momentum=0.9, weight_decay=1e-4)
C.apply(weight_init)
if use_gyd:
    G_yd_optim = torch.optim.SGD(G_yd.parameters(), lr_d, momentum=0.9, weight_decay=1e-4)
    G_yd.apply(weight_init)
if use_dec:
    Dec_optim = torch.optim.SGD(Dec.parameters(), lr_dec, momentum=0.9, weight_decay=1e-4)
    Dec.apply(weight_init)

if use_tensorboard:
    logger = Log.Logger(tb_path)
lambda1 = lambda epo: 1/math.pow(1+10*epo/epoch, 0.75)
G_f_scheduler = LambdaLR(G_f_optim, lr_lambda=lambda1)
G_d_scheduler = LambdaLR(G_d_optim, lr_lambda=lambda1)
if use_gyd:
    G_yd_scheduler = LambdaLR(G_yd_optim, lr_lambda=lambda1)
C_scheduler = LambdaLR(C_optim, lr_lambda=lambda1)

s_train_iter = s_train_dtl.__iter__()
t_train_iter = t_train_dtl.__iter__()
logs, logt = [], []

for step in range(epoch):
    # print("step:  ", step)
    G_f_scheduler.step()
    G_d_scheduler.step()
    C_scheduler.step()
    if use_gyd:
        G_yd_scheduler.step()
    lambda_d = 0.6
    # lambda_g = 0.25* (2 / (1 + math.exp(-step / epoch)) - 1)  # 0.1似乎太小 ->0.25
    lambda_g = 0.25
    try:
        x_s, y_s = next(s_train_iter)
    except StopIteration:
        s_train_iter = s_train_dtl.__iter__()
        x_s, y_s = next(s_train_iter)
    try:
        x_t, y_t = next(t_train_iter)
    except StopIteration:
        t_train_iter = t_train_dtl.__iter__()
        x_t, y_t = next(t_train_iter)
    if use_gpu:
        x_s, x_t, y_s, y_t = Variable(x_s.cuda()), Variable(x_t.cuda()), Variable(y_s.cuda()), Variable(y_t.cuda())
    else:
        x_s, x_t, y_s, y_t = Variable(x_s), Variable(x_t), Variable(y_s), Variable(y_t)
    source_label.data.resize_(x_s.size()[0]).fill_(1)
    target_label.data.resize_(x_t.size()[0]).fill_(0)

    # ------ train D and C ------
    C.zero_grad()
    feature_s = G_f(x_s)
    feature_t = G_f(x_t)
    feature_s_d = feature_s.detach()
    feature_t_d = feature_t.detach()
    # C
    out_s = C(feature_s_d)
    # Gd
    dm_s = G_d(feature_s_d)
    dm_t = G_d(feature_t_d)
    # --- Loss ---
    s_clf_loss = clf_loss(out_s, y_s)
    s_clf_loss.backward()
    C_optim.step()
    if use_gd:
        G_d.zero_grad()
        # original or least square GAN
        gd_d_loss = lambda_d * (domain_clf_loss(dm_s, source_label) + domain_clf_loss(dm_t, target_label))
        gd_d_loss.backward()
        G_d_optim.step()
    if use_gyd:
        pred_s = softmax(out_s).detach()
        pred_t = softmax(C(feature_t_d)).detach()
        G_yd.zero_grad()
        ydm_s = G_yd(torch.cat([feature_s_d, pred_s], 1))
        ydm_t = G_yd(torch.cat([feature_t_d, pred_t], 1))
        gyd_d_loss = lambda_d*(domain_clf_loss(ydm_s, source_label) + domain_clf_loss(ydm_t, target_label))
        gyd_d_loss.backward()
        G_yd_optim.step()
    if use_dec:
        # Dec
        Dec.zero_grad()
        res = Dec(feature_t_d)
        dec_loss = lambda_dec*sc_loss(res, x_s)
        dec_loss.backward()
        Dec_optim.step()

    # ------ train G ------
    source_label.data.fill_(0)
    target_label.data.fill_(1)
    G_f.zero_grad()
    # C
    out_s = C(feature_s)
    pred_s = softmax(out_s).detach()
    # Gd
    if use_gd or use_gyd:
        dm_s = G_d(feature_s)
        dm_t = G_d(feature_t)
    # --- Loss ---
    # pay attention to, this is the second original loss of GAN, can be seen as domain confusion loss
    s_clf_loss = clf_loss(out_s, y_s)
    total_loss = s_clf_loss
    if use_gd:
        gd_g_loss = lambda_g * (lambda_cnf*domain_clf_loss(dm_s, source_label) + domain_clf_loss(dm_t, target_label))
        total_loss = gd_g_loss + total_loss
    if use_gyd:
        pred_t = softmax(C(feature_t.detach()))
        G_yd.zero_grad()
        ydm_s = G_yd(torch.cat([feature_s, pred_s], 1))
        ydm_t = G_yd(torch.cat([feature_t, pred_t], 1))
        gyd_g_loss = lambda_g*(lambda_cnf*domain_clf_loss(ydm_s, source_label) + domain_clf_loss(ydm_t, target_label))
        total_loss = total_loss + gyd_g_loss
    if use_dec:
        # Dec
        res = Dec(feature_t)
        dec_loss = lambda_dec * sc_loss(res, x_s)
        total_loss = total_loss + dec_loss
    total_loss.backward()
    G_f_optim.step()

    # ------ show ------
    if step%show_step == 0:
        info = {
            's_clf_loss': v2np(s_clf_loss)[0],
        }
        print(
            "***Train***, At step {}, the source classification loss is {}".format(step, info['s_clf_loss']),
        )
        # if step >= 500:
        #     print(feature_s_d[0], out_s[0], G_f.extra.parameters())
        if use_gd:
            info['Gd_loss_d'] = v2np(gd_d_loss)[0]
            info['Gd_loss_g'] = v2np(gd_g_loss)[0]
            print(
                "Gd Loss is d_step:{}  g_step{}".format(info['Gd_loss_d'], info['Gd_loss_g'])
            )
        if use_gyd:
            info['Gyd_loss_d'] = v2np(gyd_d_loss)[0]
            info['Gyd_loss_g'] = v2np(gyd_g_loss)[0]
            print("Gyd Loss is d_step:{}   g_step:{}".format(info['Gyd_loss_d'], info['Gyd_loss_g']))
        if use_dec:
            info['Dec_loss'] = v2np(dec_loss)[0]
            print("Dec Loss is{}".format(info['Dec_loss']))
        if use_tensorboard:
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)
            for tag, value in G_f.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, v2np(value), step)
                logger.histo_summary(tag+'/grad', v2np(value.grad), step)

        # ------ test ------
        if step % test_step == 0:
            t_info = {}
            G_f.eval()
            C.eval()
            log_acc = 0
            for t_step, (x_ts, y_ts) in enumerate(s_test_dtl):
                if use_gpu:
                    x_ts, y_ts = Variable(x_ts.cuda()), Variable(y_ts.cuda())
                else:
                    x_ts, y_ts = Variable(x_ts), Variable(y_ts)
                x_ts.volatile = True  # that is important, backward() can clear the graph
                feature = G_f(x_ts)
                ys_prob = C(feature)
                _, ys_pred = torch.max(ys_prob, 1)
                log_acc = log_acc + v2np((ys_pred == y_ts).float().sum())[0]
            log_acc = (log_acc/len(s_test_dtl.dataset))*100
            logs.append(log_acc)
            t_info['s_acc'] = log_acc
            print(
                "***Test*** At step {}".format(step), "the source accuracy is {:.2f}%".
                    format(log_acc)
            )
            log_acc = 0
            for t_step, (x_tt, y_tt) in enumerate(t_test_dtl):
                if use_gpu:
                    x_tt, y_tt = Variable(x_tt.cuda()), Variable(y_tt.cuda())
                else:
                    x_tt, y_tt = Variable(x_tt), Variable(y_tt)
                x_tt.volatile = True
                feature = G_f(x_tt)
                yt_prob = C(feature)
                _, yt_pred = torch.max(yt_prob, 1)
                log_acc = log_acc + v2np((yt_pred == y_tt).float().sum())[0]
            log_acc = (log_acc/len(t_test_dtl.dataset))*100
            logt.append(log_acc)
            t_info['t_acc'] = log_acc
            print(
                "the target accuracy is {: .2f}%".
                    format(log_acc)
            )
            if use_tensorboard:
                for tag, value in t_info.items():
                    logger.scalar_summary(tag, value, step)
            G_f.train()
            C.train()






