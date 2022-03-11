from .base import KnowledgeDistillation
import torchextractor as tx
import torch
import torch.nn as nn
import torch.nn.functional as F



class ABF(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1) 

    
    def forward(self, x, y = None, shape = None):
        n, _, h, w = x.shape
        x = self.conv1(x)
        if self.att_conv is not None:
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode = "nearest")
            z = torch.cat([x, y], dim = 1)

            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        y =  self.conv2(x)
        return y, x


class ReviewKD(KnowledgeDistillation):

    def __init__(self, 
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        list_named_features_s = None,
        list_named_features_t = None
    ):


        super(ReviewKD, self).__init__(teacher_model, student_model, train_loader, val_loader, optimizer, loss_fn, temp, distil_weight, device)
        self.list_named_features_s = list_named_features_s
        self.list_named_features_t = list_named_features_t



    def train_epoch(self):

        self.teacher_model.eval()
        self.student_model.train()

        self.teacher_model = tx.Extractor(self.teacher_model, self.list_named_features_t)
        self.student_model = tx.Extractor(self.student_model, self.list_named_features_s)
        for (inputs, targets) in self.train_loader:
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            

    def eval_epoch(self):

        pass

    def calc_feature_loss(self, features_s, features_t):
        loss_all = 0.0
        for fs, ft in zip(features_s, features_t):
            n, c, h, w  = fs.shape
            loss = F.mse_loss(fs, ft, reduction = 'mean')
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))

                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction = "mean") * cnt
                tot += cnt
            loss = loss / tot
        loss_all = loss_all + loss


        


