import torch
import subprocess as sub
from torch import optim
#from torch.autograd import Variable
import torch.nn.functional as F
#import loadData3 as loadData
#import loadData2_latest as loadData
#import loadData
import numpy as np
import time
import os
#from LogMetric import Logger
import argparse
#from models.encoder_plus import Encoder
#from models.encoder import Encoder
#from models.encoder_bn_relu import Encoder
from models.encoder_vgg import Encoder
from models.decoder import Decoder
from models.attention import locationAttention as Attention
#from models.attention import TroAttention as Attention
from models.seq2seq import Seq2Seq
from utils import visualizeAttn, writePredict, writeLoss, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP, WORD_LEVEL, load_data_func, tokens, GT_TE

parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

#torch.cuda.set_device(1)
device = torch.device('cuda')

NUM_THREAD = 2

LABEL_SMOOTH = True

Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 32
learning_rate = 2 * 1e-4
#lr_milestone = [30, 50, 70, 90, 120]
#lr_milestone = [20, 40, 60, 80, 100]
#lr_milestone = [15, 25, 35, 45, 55, 65]
#lr_milestone = [30, 40, 50, 60, 70]
#lr_milestone = [30, 40, 60, 80, 100]
lr_milestone = [20, 40, 60, 80, 100]
#lr_milestone = [20, 40, 46, 60, 80, 100]

lr_gamma = 0.5

START_TEST = 1e4 # 1e4: never run test 0: run test from beginning
FREEZE = False
freeze_milestone = [65, 90]
EARLY_STOP_EPOCH = 30 # None: no early stopping
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512 # model/encoder.py SUM_UP=False: enc:dec = 1:2  SUM_UP=True: enc:dec = 1:1
CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
CurriculumModelID = args.start_epoch
#CurriculumModelID = -1 # < 0: do not use curriculumLearning, train from scratch
#CurriculumModelID = 170 # 'save_weights/seq2seq-170.model.backup'
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5
TEACHER_FORCING = False
MODEL_SAVE_EPOCH = 1

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        #self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.criterion = torch.nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.detach() == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        if true_dist.requires_grad:
            print('Error! true_dist should not requires_grad!')
        return self.criterion(x, true_dist)

log_softmax = torch.nn.LogSoftmax(dim=-1)
crit = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], 0.4)
# predict and gt follow the same shape of cross_entropy
# predict: 704, 83   gt: 704
def loss_label_smoothing(predict, gt):
    def smoothlabel_torch(x, amount=0.25, variance=5):
        mu = amount/x.shape[0]
        sigma = mu/variance
        noise = np.random.normal(mu, sigma, x.shape).astype('float32')
        smoothed = x*torch.from_numpy(1-noise.sum(1)).view(-1, 1).to(device) + torch.from_numpy(noise).to(device)
        return smoothed

    def one_hot(src): # src: torch.cuda.LongTensor
        ones = torch.eye(vocab_size).to(device)
        return ones.index_select(0, src)

    gt_local = one_hot(gt.detach())
    gt_local = smoothlabel_torch(gt_local)
    loss_f = torch.nn.BCEWithLogitsLoss()
    if not gt_local.requires_grad:
        print('Error! gt_local should have requires_grad=True')
    res_loss = loss_f(predict, gt_local)
    return res_loss

def teacher_force_func(epoch):
    if epoch < 50:
        teacher_rate = 0.5
    elif epoch < 150:
        teacher_rate = (50 - (epoch-50)//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate

def teacher_force_func_2(epoch):
    if epoch < 200:
        teacher_rate = (100 - epoch//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate


def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return train_loader, valid_loader, test_loader

def test_data_loader_batch(batch_size_nuevo):
    _, _, data_test = load_data_func()
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size_nuevo, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return test_loader

def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        idx, img, img_width, label = batch[i]
        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    return train_index, train_in, train_in_len, train_out

def train(train_loader, seq2seq, opt, teacher_rate, epoch):
    seq2seq.train()
    total_loss = 0
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(train_loader):
        #train_in = train_in.unsqueeze(1)
        train_in, train_out = train_in.requires_grad_().to(device), train_out.to(device)
        if not train_in.requires_grad:
            print('ERROR! train_in should have requires_grad=True')
        output, attn_weights = seq2seq(train_in, train_out, train_in_len, teacher_rate=teacher_rate, train=True) # (100-1, 32, 62+1)
        batch_count_n = writePredict(epoch, train_index, output, 'train')
        train_label = train_out.permute(1, 0)[1:].reshape(-1)#remove<GO>
        output_l = output.reshape(-1, vocab_size) # remove last <EOS>

        if VISUALIZE_TRAIN:
            if 'e02-074-03-00,191' in train_index:
                b = train_index.tolist().index('e02-074-03-00,191')
                visualizeAttn(train_in.detach()[b,0], train_in_len[0], [j[b] for j in attn_weights], epoch, batch_count_n[b], 'train_e02-074-03-00')

        #loss = F.cross_entropy(output_l.view(-1, vocab_size),
        #                       train_label, ignore_index=tokens['PAD_TOKEN'])
        #loss = loss_label_smoothing(output_l.view(-1, vocab_size), train_label)
        if LABEL_SMOOTH:
            loss = crit(log_softmax(output_l.reshape(-1, vocab_size)), train_label)
        else:
            loss = F.cross_entropy(output_l.reshape(-1, vocab_size),
                               train_label, ignore_index=tokens['PAD_TOKEN'])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    total_loss /= (num+1)
    return total_loss

def valid(valid_loader, seq2seq, epoch):
    seq2seq.eval()
    total_loss_t = 0
    with torch.no_grad():
        for num, (test_index, test_in, test_in_len, test_out) in enumerate(valid_loader):
            #test_in = test_in.unsqueeze(1)
            test_in, test_out = test_in.to(device), test_out.to(device)
            if test_in.requires_grad or test_out.requires_grad:
                print('ERROR! test_in, test_out should have requires_grad=False')
            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            batch_count_n = writePredict(epoch, test_index, output_t, 'valid')
            test_label = test_out.permute(1, 0)[1:].reshape(-1)
            #loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
            #                         test_label, ignore_index=tokens['PAD_TOKEN'])
            #loss_t = loss_label_smoothing(output_t.view(-1, vocab_size), test_label)
            if LABEL_SMOOTH:
                loss_t = crit(log_softmax(output_t.reshape(-1, vocab_size)), test_label)
            else:
                loss_t = F.cross_entropy(output_t.reshape(-1, vocab_size),
                                     test_label, ignore_index=tokens['PAD_TOKEN'])

            total_loss_t += loss_t.item()

            if 'n04-015-00-01,171' in test_index:
                b = test_index.tolist().index('n04-015-00-01,171')
                visualizeAttn(test_in.detach()[b,0], test_in_len[0], [j[b] for j in attn_weights_t], epoch, batch_count_n[b], 'valid_n04-015-00-01')
        total_loss_t /= (num+1)
    return total_loss_t

def test(test_loader, modelID, showAttn=True):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).to(device)
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).to(device)
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).to(device)
    model_file = 'save_weights/seq2seq-' + str(modelID) +'.model'
    print('Loading ' + model_file)
    seq2seq.load_state_dict(torch.load(model_file)) #load

    seq2seq.eval()
    total_loss_t = 0
    start_t = time.time()
    with torch.no_grad():
        for num, (test_index, test_in, test_in_len, test_out) in enumerate(test_loader):
            #test_in = test_in.unsqueeze(1)
            test_in, test_out = test_in.to(device), test_out.to(device)
            if test_in.requires_grad or test_out.requires_grad:
                print('ERROR! test_in, test_out should have requires_grad=False')
            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            batch_count_n = writePredict(modelID, test_index, output_t, 'test')
            test_label = test_out.permute(1, 0)[1:].reshape(-1)
            #loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
            #                        test_label, ignore_index=tokens['PAD_TOKEN'])
            #loss_t = loss_label_smoothing(output_t.view(-1, vocab_size), test_label)
            if LABEL_SMOOTH:
                loss_t = crit(log_softmax(output_t.reshape(-1, vocab_size)), test_label)
            else:
                loss_t = F.cross_entropy(output_t.reshape(-1, vocab_size),
                                    test_label, ignore_index=tokens['PAD_TOKEN'])

            total_loss_t += loss_t.item()

            if showAttn:
                global_index_t = 0
                for t_idx, t_in in zip(test_index, test_in):
                    visualizeAttn(t_in.detach()[0], test_in_len[0], [j[global_index_t] for j in attn_weights_t], modelID, batch_count_n[global_index_t], 'test_'+t_idx.split(',')[0])
                    global_index_t += 1

        total_loss_t /= (num+1)
        writeLoss(total_loss_t, 'test')
        print('    TEST loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

def main(train_loader, valid_loader, test_loader):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).to(device)
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).to(device)
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).to(device)
    if CurriculumModelID > 0:
        model_file = 'save_weights/seq2seq-' + str(CurriculumModelID) +'.model'
        #model_file = 'save_weights/words/seq2seq-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        seq2seq.load_state_dict(torch.load(model_file)) #load
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    #opt = optim.SGD(seq2seq.parameters(), lr=learning_rate, momentum=0.9)
    #opt = optim.RMSprop(seq2seq.parameters(), lr=learning_rate, momentum=0.9)

    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=1)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)
    epochs = 5000000
    if EARLY_STOP_EPOCH is not None:
        min_loss = 1e3
        min_loss_index = 0
        min_loss_count = 0

    if CurriculumModelID > 0 and WORD_LEVEL:
        start_epoch = CurriculumModelID + 1
        for i in range(start_epoch):
            scheduler.step()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        teacher_rate = teacher_force_func(epoch) if TEACHER_FORCING else False
        start = time.time()
        loss = train(train_loader, seq2seq, opt, teacher_rate, epoch)
        writeLoss(loss, 'train')
        print('epoch %d/%d, loss=%.3f, lr=%.8f, teacher_rate=%.3f, time=%.3f' % (epoch, epochs, loss, lr, teacher_rate, time.time()-start))

        if epoch%MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(seq2seq.state_dict(), folder_weights+'/seq2seq-%d.model'%epoch)

        start_v = time.time()
        loss_v = valid(valid_loader, seq2seq, epoch)
        writeLoss(loss_v, 'valid')
        print('  Valid loss=%.3f, time=%.3f' % (loss_v, time.time()-start_v))

        test(test_loader, epoch, False) #~~~~~~

        if EARLY_STOP_EPOCH is not None:
            gt = GT_TE
            decoded = 'pred_logs/test_predict_seq.'+str(epoch)+'.log'
            res_cer = sub.Popen(['./tasas_cer.sh', gt, decoded], stdout=sub.PIPE)
            res_cer = res_cer.stdout.read().decode('utf8')
            loss_v = float(res_cer)/100
            if loss_v < min_loss:
                min_loss = loss_v
                min_loss_index = epoch
                min_loss_count = 0
            else:
                min_loss_count += 1
            if min_loss_count >= EARLY_STOP_EPOCH:
                print('Early Stopping at: %d. Best epoch is: %d' % (epoch, min_loss_index))
                return min_loss_index

if __name__ == '__main__':
    print(time.ctime())
    train_loader, valid_loader, test_loader = all_data_loader()
    mejorModelID = main(train_loader, valid_loader, test_loader)
    #test(test_loader, mejorModelID, True)
    #os.system('./test.sh '+str(mejorModelID))
    print(time.ctime())
