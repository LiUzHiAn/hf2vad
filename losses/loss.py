import torch
import torch.nn as nn


def latent_kl(prior_mean, posterior_mean):
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])

    return torch.mean(kl)


def aggregate_kl_loss(prior_means, posterior_means):
    kl_stages = []
    for p, q in zip(list(prior_means.values()), list(posterior_means.values())):
        kl_stages.append(latent_kl(p, q).unsqueeze(dim=-1))

    kl_stages = torch.cat(kl_stages, dim=-1)
    kl_loss = torch.sum(kl_stages, dim=-1)
    return kl_loss


class Intensity_Loss(nn.Module):
    def __init__(self, l_num):
        super(Intensity_Loss, self).__init__()
        self.l_num = l_num

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** self.l_num))


class Gradient_Loss(nn.Module):
    def __init__(self, alpha, channels, device):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.device = device
        filter = torch.FloatTensor([[-1., 1.]]).to(device)

        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))

        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)


class Entropy_Loss(nn.Module):
    def __init__(self):
        super(Entropy_Loss, self).__init__()

    def forward(self, x):
        eps = 1e-20
        tmp = torch.sum((-x) * torch.log(x + eps), dim=-1)
        return torch.mean(tmp)
