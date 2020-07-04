import torch
from torch import nn
#from torch.autograd import Variable
import random

print_shape_flag = True

cpu = torch.device('cpu')
cuda = torch.device('cuda')

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_max_len, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size

    # src: Variable
    # tar: Variable
    def forward(self, src, tar, src_len, teacher_rate, train=True):
        tar = tar.permute(1, 0) # time_s, batch
        batch_size = src.size(0)
        #max_len = tar.size(0) # <go> true_value <end>
        #outputs = Variable(torch.zeros(self.output_max_len-1, batch_size, self.vocab_size), requires_grad=True) # (14, 32, 62) not save the first <GO>
        outputs = torch.zeros(self.output_max_len-1, batch_size, self.vocab_size, requires_grad=True).to(cuda) # (14, 32, 62) not save the first <GO>
        #src = Variable(src)
        out_enc, hidden_enc = self.encoder(src, src_len)
        # t,b,f    layers, b,f
        global print_shape_flag
        if print_shape_flag:
            print('First batch shape: (The shape of batches are not same)')
            print(out_enc.shape, self.output_max_len)
            print_shape_flag = False

        output = self.one_hot(tar[0].detach()).requires_grad_()
        if not output.requires_grad:
            print("ERROR! output should have requires_grad=True")
        attns = []

        hidden = hidden_enc
        #hidden = hidden_enc.unsqueeze(0) # 1, batch, hidden_size
        #init_hidden_dec = [hidden] * self.decoder.n_layers
        #hidden = torch.cat(init_hidden_dec, dim=0)
        attn_weights = torch.zeros(out_enc.shape[1], out_enc.shape[0], requires_grad=True).to(cuda) # b, t

        for t in range(0, self.output_max_len-1): # max_len: groundtruth + <END>
            teacher_force_rate = random.random() < teacher_rate
            output, hidden, attn_weights = self.decoder(
                    output, hidden, out_enc, src_len, attn_weights)
            outputs[t] = output
            #top1 = output.data.topk(1)[1].squeeze()
            output = self.one_hot(tar[t+1].detach()) if train and teacher_force_rate else output.detach()
            attns.append(attn_weights.detach().to(cpu)) # [(32, 55), ...]
        return outputs, attns

    def one_hot(self, src): # src: torch.cuda.LongTensor
        ones = torch.eye(self.vocab_size).to(cuda)
        return ones.index_select(0, src)
