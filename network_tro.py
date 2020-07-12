import torch
import torch.nn as nn
from load_data import vocab_size, IMG_WIDTH, OUTPUT_MAX_LEN
from modules_tro import GenModel_FC, DisModel, WriterClaModel, RecModel, write_image
from loss_tro import recon_criterion, crit, log_softmax
import numpy as np

w_dis = 1.
w_cla = 1.
w_l1 = 0.
w_rec = 1.

gpu = torch.device('cuda')

class ConTranModel(nn.Module):
    def __init__(self, num_writers, show_iter_num, oov):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC(OUTPUT_MAX_LEN).to(gpu)
        self.cla = WriterClaModel(num_writers).to(gpu)
        self.dis = DisModel().to(gpu)
        self.rec = RecModel(pretrain=False).to(gpu)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov

    def forward(self, train_data_list, epoch, mode, cer_func=None):
        tr_domain, tr_wid, tr_idx, tr_img, tr_img_width, tr_label, img_xt, label_xt, label_xt_swap = train_data_list
        tr_wid = tr_wid.to(gpu)
        tr_img = tr_img.to(gpu)
        tr_img_width = tr_img_width.to(gpu)
        tr_label = tr_label.to(gpu)
        img_xt = img_xt.to(gpu)
        label_xt = label_xt.to(gpu)
        label_xt_swap = label_xt_swap.to(gpu)
        batch_size = tr_domain.shape[0]

        if mode == 'rec_update':
            tr_img_rec = tr_img[:, 0:1, :, :] # 8,50,64,200 choose one channel 8,1,64,200
            tr_img_rec = tr_img_rec.requires_grad_()
            tr_label_rec = tr_label[:, 0, :] # 8,50,10 choose one channel 8,10
            pred_xt_tr = self.rec(tr_img_rec, tr_label_rec, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            tr_label_rec2 = tr_label_rec[:, 1:] # remove <GO>
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1,vocab_size)), tr_label_rec2.reshape(-1))
            cer_func.add(pred_xt_tr, tr_label_rec2)
            l_rec_tr.backward()
            return l_rec_tr

        elif mode =='cla_update':
            tr_img_rec = tr_img[:, 0:1, :, :] # 8,50,64,200 choose one channel 8,1,64,200
            tr_img_rec = tr_img_rec.requires_grad_()
            l_cla_tr = self.cla(tr_img_rec, tr_wid)
            l_cla_tr.backward()
            return l_cla_tr

        elif mode == 'gen_update':
            self.iter_num += 1
            '''dis loss'''
            f_xs = self.gen.enc_image(tr_img) # b,512,8,27
            f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape) # b,4096  b,512,8,27
            f_mix = self.gen.mix(f_xs, f_embed)

            xg = self.gen.decode(f_mix, f_xt)  # translation b,1,64,128
            l_dis_ori = self.dis.calc_gen_loss(xg)

            # '''poco modi -> swap char'''
            f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
            f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
            xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)  # translation b,1,64,128
            l_dis_swap = self.dis.calc_gen_loss(xg_swap)
            l_dis = (l_dis_ori + l_dis_swap) / 2.

            '''writer classifier loss'''
            l_cla_ori = self.cla(xg, tr_wid)
            l_cla_swap = self.cla(xg_swap, tr_wid)
            l_cla = (l_cla_ori + l_cla_swap) / 2.

            '''l1 loss'''
            if self.oov:
                l_l1 = torch.tensor(0.).to(gpu)
            else:
                l_l1 = recon_criterion(xg, img_xt)

            '''rec loss'''
            cer_te, cer_te2 = cer_func
            pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            label_xt2 = label_xt[:, 1:] # remove <GO>
            label_xt2_swap = label_xt_swap[:, 1:] # remove <GO>
            l_rec_ori = crit(log_softmax(pred_xt.reshape(-1,vocab_size)), label_xt2.reshape(-1))
            l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1,vocab_size)), label_xt2_swap.reshape(-1))
            cer_te.add(pred_xt, label_xt2)
            cer_te2.add(pred_xt_swap, label_xt2_swap)
            l_rec = (l_rec_ori + l_rec_swap) / 2.
            '''fin'''
            l_total = w_dis * l_dis + w_cla * l_cla + w_l1 * l_l1 + w_rec * l_rec
            l_total.backward()
            return l_total, l_dis, l_cla, l_l1, l_rec

        elif mode == 'dis_update':
            sample_img1 = tr_img[:,0:1,:,:]
            sample_img2 = tr_img[:,1:2,:,:]
            sample_img1.requires_grad_()
            sample_img2.requires_grad_()
            l_real1 = self.dis.calc_dis_real_loss(sample_img1)
            l_real2 = self.dis.calc_dis_real_loss(sample_img2)
            l_real = (l_real1 + l_real2) / 2.
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img)
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xs, f_embed)
                xg = self.gen.decode(f_mix, f_xt)
                # swap tambien
                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)

            l_fake_ori = self.dis.calc_dis_fake_loss(xg)
            l_fake_swap = self.dis.calc_dis_fake_loss(xg_swap)
            l_fake = (l_fake_ori + l_fake_swap) / 2.
            l_fake.backward()

            l_total = l_real + l_fake
            '''write images'''
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                    pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                write_image(xg, pred_xt, img_xt, label_xt, tr_img, xg_swap, pred_xt_swap, label_xt_swap, 'epoch_'+str(epoch)+'-'+str(self.iter_num))
            return l_total

        elif mode =='eval':
            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img)
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xs, f_embed)
                xg = self.gen.decode(f_mix, f_xt)
                # second oov word
                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)
                '''write images'''
                pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                write_image(xg, pred_xt, img_xt, label_xt, tr_img, xg_swap, pred_xt_swap, label_xt_swap, 'eval_'+str(epoch)+'-'+str(self.iter_num))
                self.iter_num += 1
                '''dis loss'''
                l_dis_ori = self.dis.calc_gen_loss(xg)
                l_dis_swap = self.dis.calc_gen_loss(xg_swap)
                l_dis = (l_dis_ori + l_dis_swap) / 2.

                '''rec loss'''
                cer_te, cer_te2 = cer_func
                label_xt2 = label_xt[:, 1:] # remove <GO>
                label_xt2_swap = label_xt_swap[:, 1:] # remove <GO>
                l_rec_ori = crit(log_softmax(pred_xt.reshape(-1,vocab_size)), label_xt2.reshape(-1))
                l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1,vocab_size)), label_xt2_swap.reshape(-1))
                cer_te.add(pred_xt, label_xt2)
                cer_te2.add(pred_xt_swap, label_xt2_swap)
                l_rec = (l_rec_ori + l_rec_swap) / 2.

                '''writer classifier loss'''
                l_cla_ori = self.cla(xg, tr_wid)
                l_cla_swap = self.cla(xg_swap, tr_wid)
                l_cla = (l_cla_ori + l_cla_swap) / 2.

            return l_dis, l_cla, l_rec
