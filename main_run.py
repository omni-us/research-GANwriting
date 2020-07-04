import os
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from load_data import NUM_WRITERS
from network_tro import ConTranModel
from load_data import loadData as load_data_func
from loss_tro import CER

parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

gpu = torch.device('cuda')

OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 200
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 8
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5
lr_cla = 1 * 1e-5

CurriculumModelID = args.start_epoch


def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return train_loader, test_loader


def sort_batch(batch):
    train_domain = list()
    train_wid = list()
    train_idx = list()
    train_img = list()
    train_img_width = list()
    train_label = list()
    img_xts = list()
    label_xts = list()
    label_xts_swap = list()
    for domain, wid, idx, img, img_width, label, img_xt, label_xt, label_xt_swap in batch:
        if wid >= NUM_WRITERS:
            print('error!')
        train_domain.append(domain)
        train_wid.append(wid)
        train_idx.append(idx)
        train_img.append(img)
        train_img_width.append(img_width)
        train_label.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    train_domain = np.array(train_domain)
    train_idx = np.array(train_idx)
    train_wid = np.array(train_wid, dtype='int64')
    train_img = np.array(train_img, dtype='float32')
    train_img_width = np.array(train_img_width, dtype='int64')
    train_label = np.array(train_label, dtype='int64')
    img_xts = np.array(img_xts, dtype='float32')
    label_xts = np.array(label_xts, dtype='int64')
    label_xts_swap = np.array(label_xts_swap, dtype='int64')

    train_wid = torch.from_numpy(train_wid)
    train_img = torch.from_numpy(train_img)
    train_img_width = torch.from_numpy(train_img_width)
    train_label = torch.from_numpy(train_label)
    img_xts = torch.from_numpy(img_xts)
    label_xts = torch.from_numpy(label_xts)
    label_xts_swap = torch.from_numpy(label_xts_swap)

    return train_domain, train_wid, train_idx, train_img, train_img_width, train_label, img_xts, label_xts, label_xts_swap

def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_cla = list()
    loss_cla_tr = list()
    loss_l1 = list()
    loss_rec = list()
    loss_rec_tr = list()
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    for train_data_list in train_loader:
        '''rec update'''
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
        rec_opt.step()

        '''classifier update'''
        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, 'cla_update')
        cla_opt.step()

        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        dis_opt.step()

        '''gen update'''
        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_l1, l_rec = model(train_data_list, epoch, 'gen_update', [cer_te, cer_te2])
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_cla_tr.append(l_cla_tr.cpu().item())
        loss_l1.append(l_l1.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_cla = np.mean(loss_cla)
    fl_cla_tr = np.mean(loss_cla_tr)
    fl_l1 = np.mean(loss_l1)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print('epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f, l1=%.2f, cer=%.2f-%.2f-%.2f, time=%.1f' % (epoch, fl_dis_tr, fl_dis, fl_cla_tr, fl_cla, fl_rec_tr, fl_rec, fl_l1, res_cer_tr, res_cer_te, res_cer_te2, time.time()-time_s))
    return res_cer_te + res_cer_te2

def test(test_loader, epoch, modelFile_o_model):
    if type(modelFile_o_model) == str:
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model)) #load
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_cla = list()
    loss_rec = list()
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()
    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, 'eval', [cer_te, cer_te2])

        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_cla = np.mean(loss_cla)
    fl_rec = np.mean(loss_rec)

    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print('EVAL: l_dis=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f, time=%.1f' % (fl_dis, fl_cla, fl_rec, res_cer_te, res_cer_te2, time.time()-time_s))

def main(train_loader, test_loader, num_writers):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)

    if CurriculumModelID > 0:
        model_file = 'save_weights/contran-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file)) #load
        #pretrain_dict = torch.load(model_file)
        #model_dict = model.state_dict()
        #pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and not k.startswith('gen.enc_text.fc')}
        #model_dict.update(pretrain_dict)
        #model.load_state_dict(model_dict)

    dis_params = list(model.dis.parameters())
    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    cla_params = list(model.cla.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    cla_opt = optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)
    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    for epoch in range(CurriculumModelID, epochs):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights+'/contran-%d.model'%epoch)

        if epoch % EVAL_EPOCH == 0:
            test(test_loader, epoch, model)

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer:
                min_cer = cer
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print('Early stop at %d and the best epoch is %d' % (epoch, min_idx))
                model_url = 'save_weights/contran-'+str(min_idx)+'.model'
                os.system('mv '+model_url+' '+model_url+'.bak')
                os.system('rm save_weights/contran-*.model')
                break

def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-'+str(epoch)+'.model')

if __name__ == '__main__':
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader, NUM_WRITERS)
    print(time.ctime())
