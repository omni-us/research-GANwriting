import torch.utils.data as D
import cv2
import numpy as np
import random
#from torchvision import transforms
from marcal_augmentor_v4.marcal_augmentor_v4 import augmentor
#import Augmentor
#from torchsample.transforms import RangeNormalize
#import torch

WORD_LEVEL = True

VGG_NORMAL = True
HAY_THRESH = True
FLIP = False # flip the image

if WORD_LEVEL:
    OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
    IMG_WIDTH = 1011 # m01-084-07-00 max_length
    baseDir = '/home/lkang/datasets/iam_final_forms/'

    train_set = '/home/lkang/datasets/iam_final_forms/RWTH.iam_word_gt_final.train.thresh'
    valid_set = '/home/lkang/datasets/iam_final_forms/RWTH.iam_word_gt_final.valid.thresh'
    test_set = '/home/lkang/datasets/iam_final_forms/RWTH.iam_word_gt_final.test.thresh'
else:
    OUTPUT_MAX_LEN = 95 # line-level
    IMG_WIDTH = 2227 # m03-118-05.png max_length
    baseDir = '/home/lkang/datasets/iam_final_lines/'

    train_set = None
    valid_set = None
    test_set = None

IMG_HEIGHT = 64

GT_TR = train_set
GT_VA = valid_set
GT_TE = test_set

def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class IAM_words(D.Dataset):
    def __init__(self, file_label, augmentation=True, p_aug=1.):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation
        self.p_aug = p_aug
        self.transformer = augmentor

    def __getitem__(self, index):
        word = self.file_label[index]
        img, img_width = self.readImage_keepRatio(word[0], flip=FLIP)
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens)
        return word[0], img, img_width, label
        #return {'index_sa': file_name, 'input_sa': in_data, 'output_sa': out_data, 'in_len_sa': in_len, 'out_len_sa': out_data_mask}

    def __len__(self):
        return len(self.file_label)

    def readImage_keepRatio(self, file_name, flip):
        if HAY_THRESH:
            file_name, thresh = file_name.split(',')
            thresh = int(thresh)
        if WORD_LEVEL:
            subdir = 'words_from_forms/'
        else:
            subdir = 'lines/'
        url = baseDir + subdir + file_name + '.png'
        img = cv2.imread(url, 0)

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        # c04-066-01-08.png 4*3, for too small images do not augment
        img = img/255. # 0-255 -> 0-1
        if self.augmentation and (random.random() < self.p_aug): # augmentation for training data
            img_new = self.transformer(img) # img: 0-1  img_new: 0-1
            if img_new.shape[0] != 0 and img_new.shape[1] != 0:
                rate = float(IMG_HEIGHT) / img_new.shape[0]
                img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
       #     else:
       #         img = 1. - img
       # else:
       #     img = 1. - img

        if HAY_THRESH:
            #img[img>(thresh/255)] = 1.
            pass

        img = 1. - img
        img_width = img.shape[-1]

        if flip: # because of using pack_padded_sequence, first flip, then pad it
            img = np.flip(img, 1)

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            #outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')
        if VGG_NORMAL:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            outImgFinal = np.zeros([3, *outImg.shape])
            for i in range(3):
                outImgFinal[i] = (outImg - mean[i]) / std[i]
            return outImgFinal, img_width

        outImg = np.vstack([np.expand_dims(outImg, 0)] * 3) # GRAY->RGB
        return outImg, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        num = self.output_max_len - len(ll) - 2
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

def loadData():
    gt_tr = GT_TR
    gt_va = GT_VA
    gt_te = GT_TE

    with open(gt_tr, 'r') as f_tr:
        data_tr = f_tr.readlines()
        file_label_tr = [i[:-1].split(' ') for i in data_tr]

    with open(gt_va, 'r') as f_va:
        data_va = f_va.readlines()
        file_label_va = [i[:-1].split(' ') for i in data_va]

    with open(gt_te, 'r') as f_te:
        data_te = f_te.readlines()
        file_label_te = [i[:-1].split(' ') for i in data_te]

    np.random.shuffle(file_label_tr)
    data_train = IAM_words(file_label_tr, augmentation=True, p_aug=0.5)
    data_valid = IAM_words(file_label_va, augmentation=False)
    data_test = IAM_words(file_label_te, augmentation=False)
    return data_train, data_valid, data_test

if __name__ == '__main__':
    pass
