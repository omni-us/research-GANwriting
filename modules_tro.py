import numpy as np
import os
import torch
from torch import nn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seq import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import OUTPUT_MAX_LEN, IMG_HEIGHT, IMG_WIDTH, vocab_size, index2letter, num_tokens, tokens
import cv2

gpu = torch.device('cuda')

def normalize(tar):
    tar = (tar - tar.min())/(tar.max()-tar.min())
    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar

def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list

def write_image(xg, pred_label, gt_img, gt_label, tr_imgs, xg_swap, pred_label_swap, gt_label_swap, title, num_tr=2):
    folder = 'imgs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr*IMG_HEIGHT, -1)
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        tar_swap = xg_swap[i].squeeze()
        src = normalize(src)
        gt = normalize(gt)
        tar = normalize(tar)
        tar_swap = normalize(tar_swap)
        gt_text = gt_label[i].tolist()
        gt_text_swap = gt_label_swap[i].tolist()
        pred_text = pred_label[i].tolist()
        pred_text_swap = pred_label_swap[i].tolist()

        gt_text = fine(gt_text)
        gt_text_swap = fine(gt_text_swap)
        pred_text = fine(pred_text)
        pred_text_swap = fine(pred_text_swap)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x!=j, gt_text))
            gt_text_swap = list(filter(lambda x: x!=j, gt_text_swap))
            pred_text = list(filter(lambda x: x!=j, pred_text))
            pred_text_swap = list(filter(lambda x: x!=j, pred_text_swap))


        gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
        gt_text_swap = ''.join([index2letter[c-num_tokens] for c in gt_text_swap])
        pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
        pred_text_swap = ''.join([index2letter[c-num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(gt_text_img_swap, gt_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img_swap, pred_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img, gt_text_img_swap, tar_swap, pred_text_img_swap])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder+'/'+title+'.png', final_out)

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        self.n_layers = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, self.final_size, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        return out.squeeze(-1).squeeze(-1) # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def calc_dis_fake_loss(self, input_fake):
        label = torch.zeros(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        label = torch.ones(input_real.shape[0], self.final_size).to(gpu)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        label = torch.ones(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

class WriterClaModel(nn.Module):
    def __init__(self, num_writers):
        super(WriterClaModel, self).__init__()
        self.n_layers = 6
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, num_writers, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat) # b,310,1,1
        loss = self.cross_entropy(out.squeeze(-1).squeeze(-1), y)
        return loss


class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder().to(gpu)
        self.enc_text = TextEncoder_FC(text_max_len).to(gpu)
        self.dec = Decoder().to(gpu)
        self.linear_mix = nn.Linear(1024, 512)

    def decode(self, content, adain_params):
        # decode content and style codes to an image
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        feat_mix = torch.cat([feat_xs, feat_embed], dim=1) # b,1024,8,27
        f = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix(f) # b,8,27,1024->b,8,27,512
        return ff.permute(0, 3, 1, 2)

class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
                nn.Linear(text_max_len*embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, 4096)
                )
        '''embed content force'''
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        xx = self.embed(x) # b,t,embed

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1) # b,t*embed
        out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long).cuda())
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)

        return out, final_res


'''VGG19_IN tro'''
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super(RecModel, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(gpu)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, OUTPUT_MAX_LEN, vocab_size).to(gpu)
        if pretrain:
            model_file = 'recognizer/save_weights/seq2seq-72.model_5.79.bak'
            print('Loading RecModel', model_file)
            self.seq2seq.load_state_dict(torch.load(model_file))

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img,img,img], dim=1) # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label, img_width, teacher_rate=False, train=False)
        return output.permute(1, 0, 2) # t,b,83->b,t,83


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
