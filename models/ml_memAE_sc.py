import math
from models.basic_modules import *


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M]
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        # if use hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)


class ML_MemAE_SC(nn.Module):
    def __init__(self, num_in_ch, seq_len, features_root,
                 num_slots, shrink_thres,
                 mem_usage, skip_ops):
        super(ML_MemAE_SC, self).__init__()
        self.num_in_ch = num_in_ch
        self.seq_len = seq_len
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops

        self.in_conv = inconv(num_in_ch * seq_len, features_root)
        self.down_1 = down(features_root, features_root * 2)
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)

        # memory modules
        self.mem1 = Memory(num_slots=self.num_slots, slot_dim=features_root * 2 * 16 * 16,
                           shrink_thres=self.shrink_thres) if self.mem_usage[1] else None
        self.mem2 = Memory(num_slots=self.num_slots, slot_dim=features_root * 4 * 8 * 8,
                           shrink_thres=self.shrink_thres) if self.mem_usage[2] else None
        self.mem3 = Memory(num_slots=self.num_slots, slot_dim=features_root * 8 * 4 * 4,
                           shrink_thres=self.shrink_thres) if self.mem_usage[3] else None

        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_in_ch * seq_len)

    def forward(self, x):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        x0 = self.in_conv(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        if self.mem_usage[3]:
            # flatten [bs,C,H,W] --> [bs,C*H*W]
            bs, C, H, W = x3.shape
            x3 = x3.view(bs, -1)
            mem3_out = self.mem3(x3)
            x3 = mem3_out["out"]
            # attention weight size [bs,N], N is num_slots
            att_weight3 = mem3_out["att_weight"]
            # unflatten
            x3 = x3.view(bs, C, H, W)

        recon = self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None)

        if self.mem_usage[2]:
            # pass through memory again
            bs, C, H, W = recon.shape
            recon = recon.view(bs, -1)
            mem2_out = self.mem2(recon)
            recon = mem2_out["out"]
            att_weight2 = mem2_out["att_weight"]
            recon = recon.view(bs, C, H, W)

        recon = self.up_2(recon, x1 if self.skip_ops[-2] != "none" else None)

        if self.mem_usage[1]:
            # pass through memory again
            bs, C, H, W = recon.shape
            recon = recon.view(bs, -1)
            mem1_out = self.mem1(recon)
            recon = mem1_out["out"]
            att_weight1 = mem1_out["att_weight"]
            recon = recon.view(bs, C, H, W)

        recon = self.up_1(recon, x0 if self.skip_ops[-3] != "none" else None)
        recon = self.out_conv(recon)

        if self.num_mem == 3:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=att_weight1)
        elif self.num_mem == 2:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2,
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        elif self.num_mem == 1:
            outs = dict(recon=recon, att_weight3=att_weight3,
                        att_weight2=torch.zeros_like(att_weight3),
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        return outs


if __name__ == '__main__':
    model = ML_MemAE_SC(num_in_ch=2, seq_len=1, features_root=32, num_slots=2000, shrink_thres=1 / 2000,
                        mem_usage=[False, True, True, True], skip_ops=["none", "concat", "concat"])
    dummy_x = torch.rand(4, 2, 32, 32)
    dummy_out = model(dummy_x)
    print(-1)
