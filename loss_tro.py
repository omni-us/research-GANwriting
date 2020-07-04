import torch
import Levenshtein as Lev
from load_data import vocab_size, tokens, num_tokens, index2letter

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
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


def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list

class CER():
    def __init__(self):
        self.ed = 0
        self.len = 0

    def add(self, pred, gt):
        pred_label = torch.topk(pred, 1, dim=-1)[1].squeeze(-1) # b,t,83->b,t,1->b,t
        pred_label = pred_label.cpu().numpy()
        batch_size = pred_label.shape[0]
        eds = list()
        lens = list()
        for i in range(batch_size):
            pred_text = pred_label[i].tolist()
            gt_text = gt[i].cpu().numpy().tolist()

            gt_text = fine(gt_text)
            pred_text = fine(pred_text)
            for j in range(num_tokens):
                gt_text = list(filter(lambda x: x!=j, gt_text))
                pred_text = list(filter(lambda x: x!=j, pred_text))
            gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
            pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
            ed_value = Lev.distance(pred_text, gt_text)
            eds.append(ed_value)
            lens.append(len(gt_text))
        self.ed += sum(eds)
        self.len += sum(lens)

    def fin(self):
        return 100 * (self.ed / self.len)

