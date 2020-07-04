import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
from modules_tro import normalize
import os


folder = 'word_ladder'
img_base = '/home/lkang/datasets/iam_final_forms/words_from_forms/'
target_file = '/home/lkang/datasets/iam_final_forms/gan.iam.test.gt.filter27'

'''data preparation'''
data_dict = dict()
with open(target_file, 'r') as _f:
    data = _f.readlines()
    data = [i.split(' ')[0] for i in data]
    data = [i.split(',') for i in data]
for wid, index in data:
    if wid in data_dict.keys():
        data_dict[wid].append(index)
    else:
        data_dict[wid] = [index]

if not os.path.exists(folder):
    os.makedirs(folder)

gpu = torch.device('cuda')

def test_writer(wid, model_file):
    def read_image(file_name, thresh=None):
        url = img_base + file_name + '.png'
        img = cv2.imread(url, 0)
        if thresh:
            #img[img>thresh] = 255
            pass

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        img = img/255. # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal

    def label_padding(labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = OUTPUT_MAX_LEN - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
        return ll

    '''data preparation'''
    imgs = [read_image(i) for i in data_dict[wid]]
    random.shuffle(imgs)
    final_imgs = imgs[:50]
    if len(final_imgs) < 50:
        while len(final_imgs) < 50:
            num_cp = 50 - len(final_imgs)
            final_imgs = final_imgs + imgs[:num_cp]

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu) # 1,50,64,216

    texts = ['three', 'threw', 'shrew', 'shred', 'sired', 'fired', 'fined', 'finer', 'fiver', 'fever', 'sever', 'seven']
    labels = torch.from_numpy(np.array([np.array(label_padding(label, num_tokens)) for label in texts])).to(gpu)

    '''model loading'''
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    print('Loading ' + model_file)
    model.load_state_dict(torch.load(model_file)) #load
    model.eval()
    num = 0
    with torch.no_grad():
        f_xs = model.gen.enc_image(imgs)
        for label in labels:
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label)
            f_mix = model.gen.mix(f_xs, f_embed)
            xg = model.gen.decode(f_mix, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            label = label.squeeze().cpu().numpy().tolist()
            pred = torch.topk(pred, 1, dim=-1)[1].squeeze()
            pred = pred.cpu().numpy().tolist()
            for j in range(num_tokens):
                label = list(filter(lambda x: x!=j, label))
                pred = list(filter(lambda x: x!=j, pred))
            label = ''.join([index2letter[c-num_tokens] for c in label])
            pred = ''.join([index2letter[c-num_tokens] for c in pred])
            ed_value = Lev.distance(pred, label)
            if ed_value <= 100:
                num += 1
                xg = xg.cpu().numpy().squeeze()
                xg = normalize(xg)
                xg = 255 - xg
                cv2.imwrite(folder+'/'+wid+'-'+str(num)+'.'+label+'-'+pred+'.png', xg)

if __name__ == '__main__':
    with open(target_file, 'r') as _f:
        data = _f.readlines()
    wids = list(set([i.split(',')[0] for i in data]))
    for wid in wids:
        test_writer(wid, 'save_weights/<your best model>')
