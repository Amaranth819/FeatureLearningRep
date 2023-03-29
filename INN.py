import torch
import torch.nn as nn
import numpy as np
import time
from utils import create_mlp, Logger
from dataset import create_GMM_dataset, onehot_to_label, visualize_GMM_data, label_to_onehot
from itertools import count
from collections import defaultdict
import matplotlib.pyplot as plt


'''
    INN model
'''
class InvertibleBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256, split_ratio = 0.5, clamp = 2, f_clamp = lambda u: 0.636 * torch.atan(u)) -> None:
        super().__init__()

        self.split_len1 = int(in_dim * split_ratio)
        self.split_len2 = in_dim - self.split_len1

        self.block1 = create_mlp(self.split_len1, self.split_len2 * 2, hidden_dim_list = [hidden_dim], act_fn = nn.ReLU)
        self.block2 = create_mlp(self.split_len2, self.split_len1 * 2, hidden_dim_list = [hidden_dim], act_fn = nn.ReLU)

        # clamp in source code, don't know why keeping it yet
        self.clamp = clamp
        self.f_clamp = f_clamp


    def forward(self, x, reverse = False):
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], -1)

        if reverse:
            # x = [v1, v2], y = [u1, u2]
            y2, j2 = self._invertible_block_2(x2, x1, reverse = True)
            y1, j1 = self._invertible_block_1(x1, y2, reverse = True)
        else:
            # x = [u1, u2], y = [v1, v2]
            y1, j1 = self._invertible_block_1(x1, x2)
            y2, j2 = self._invertible_block_2(x2, y1)

        return torch.cat([y1, y2], -1), j1 + j2


    def _invertible_block_1(self, x1, u2, reverse = False):
        st2 = self.block2(u2)
        s2, t2 = torch.split(st2, [self.split_len1, self.split_len2], -1)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim = -1)

        if reverse:
            o1 = (x1 - t2) * torch.exp(-s2)
            return o1, -j1
        else:
            o1 = x1 * torch.exp(s2) + t2
            return o1, j1


    def _invertible_block_2(self, x2, u1, reverse = False):
        tmp = self.block1(u1)
        s1, t1 = torch.chunk(tmp, 2, -1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim = -1)

        if reverse:
            o2 = (x2 - t1) * torch.exp(-s1)
            return o2, -j2
        else:
            o2 = x2 * torch.exp(s1) + t1
            return o2, j2



class PermuteRandom(nn.Module):
    def __init__(self, in_channels, seed = 123456) -> None:
        super().__init__()

        np.random.seed(seed)
        perm = np.random.permutation(in_channels)
        # perm = np.arange(in_channels)
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i
        
        self.perm = nn.Parameter(torch.from_numpy(perm), requires_grad = False)
        self.perm_inv = nn.Parameter(torch.from_numpy(perm_inv), requires_grad = False)

    
    def forward(self, x, reverse = False):
        # Only support inputs of shape [bs, n]
        return x[:, self.perm_inv] if reverse else x[:, self.perm], 0
        



class INN(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256, num_nodes = 3) -> None:
        super().__init__()

        self.nodes = []
        for _ in range(num_nodes):
            self.nodes.append(InvertibleBlock(in_dim, hidden_dim))
            self.nodes.append(PermuteRandom(in_dim))
        self.nodes = nn.ModuleList(self.nodes)


    def trainable_params(self):
        params = []
        for n in self.nodes:
            for p in n.parameters():
                if p.requires_grad:
                    params.append(p)
        return params


    def forward(self, x, reverse = False):
        for node in self.nodes[::-1 if reverse else 1]:
            x, _ = node(x, reverse)
        return x

    
    # def to(self, device):
    #     for node in self.nodes:
    #         node = node.to(device)
    #     return self


'''
    Loss function
'''
def fit_loss(pred, gt):
    return 0.5 * (pred - gt).pow(2).mean()


def MMD_loss(x, y):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

    XX, YY, XY = (torch.zeros_like(xx), torch.zeros_like(xx), torch.zeros_like(xx))

    for h in [0.05, 0.2, 0.9]:
        XX += h**2 * (h**2 + dxx).pow(-1)
        YY += h**2 * (h**2 + dyy).pow(-1)
        XY += h**2 * (h**2 + dxy).pow(-1)
    
    return torch.mean(XX + YY - 2 * XY)


'''
    Helper function
'''
def padding_func(x, pad_dim, noise_scales = 0, target_dim = -1):
    pad_shape = list(x.size())
    pad_shape[target_dim] = pad_dim
    return noise_scales * torch.randn(*pad_shape).to(x.device)


def noise_func(x, scale):
    return x + scale * torch.randn_like(x)
    



def main(epochs = 10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = './INN_exp2/'
    log = Logger(log_dir)

    # Generate dataset
    radius1 = 2.0
    radius2 = 0.15
    batch_size = 2048
    num_classes = 8
    trainset_size = 80000
    testset_size = 4000
    trainset_loader = create_GMM_dataset(trainset_size, num_classes, radius1, radius2, batch_size)
    testset_loader = create_GMM_dataset(testset_size, num_classes, radius1, radius2, testset_size)

    # Parameters
    x_ndim = 2
    y_ndim = num_classes
    z_ndim = 4 # latent dimension
    pad_ndim = 16
    zero_noise_scale = 5e-2
    label_noise_scale = 1e-1

    testset_x, testset_y = next(iter(testset_loader))
    testset_x, testset_y = testset_x.to(device), testset_y.to(device)
    testset_y_onehot = label_to_onehot(testset_y, num_classes)
    pad_testset_onehot = torch.cat([
        padding_func(testset_y_onehot, z_ndim, 1),
        padding_func(testset_y_onehot, pad_ndim - y_ndim - z_ndim, 0),
        noise_func(testset_y_onehot, label_noise_scale)
    ], -1)

    # Create network
    hidden_dim = 512
    num_nodes = 8
    model = INN(pad_ndim, hidden_dim, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.trainable_params(), lr = 1e-3, betas = (0.8, 0.9), eps = 1e-6, weight_decay = 2e-5)
    # sd = torch.load('INN_exp/INN.pkl')
    # model.load_state_dict(sd['model'])
    # optimizer.load_state_dict(sd['optimizer'])

    # print(model)
    total_steps = 0
    for e in count(1):
        for batch_idx, (x, y) in enumerate(trainset_loader):
            x, y = x.to(device), y.to(device)
            onehot_y = label_to_onehot(y, num_classes)

            # Padding
            pad1 = padding_func(x, pad_ndim - x_ndim, zero_noise_scale)
            pad_x = torch.cat([x, pad1], -1)

            noise_y = noise_func(onehot_y, label_noise_scale)
            pad2 = padding_func(onehot_y, pad_ndim - y_ndim - z_ndim, zero_noise_scale)
            z = padding_func(onehot_y, z_ndim, 1)
            pad_y = torch.cat([z, pad2, noise_y], -1)

            optimizer.zero_grad()

            # Forward
            output = model(pad_x)
            loss_y_fit_forward = 3. * fit_loss(output[:, z_ndim:], pad_y[:, z_ndim:])
            output_block_grad = torch.cat([output[:, :z_ndim], output[:, -y_ndim:].data], -1)
            pad_y_short = torch.cat([z, noise_y], -1)
            loss_yz_MMD_forward = 300 * MMD_loss(output_block_grad, pad_y_short)
            loss_forward = loss_y_fit_forward + loss_yz_MMD_forward
            loss_forward.backward()

            # Backward
            pad3 = padding_func(onehot_y, pad_ndim - y_ndim - z_ndim, zero_noise_scale)
            noise_y2 = noise_func(onehot_y, label_noise_scale)
            perturb_output_z = noise_func(output.data[:, :z_ndim], label_noise_scale)
            y_rev = torch.cat([perturb_output_z, pad3, noise_y2], -1)
            y_rev_rand = torch.cat([padding_func(onehot_y, z_ndim, 1), pad3, noise_y2], -1)
            out_rev = model(y_rev, True)
            out_rev_rand = model(y_rev_rand, True)

            loss_x_MMD_backward = 400 * min(1., 2. * 0.002**(1. - (float(e) / epochs))) * MMD_loss(out_rev_rand[:, :x_ndim], pad_x[:, :x_ndim])
            loss_x_fit_backward = 3 * fit_loss(out_rev, pad_x)
            loss_backward = loss_x_MMD_backward + loss_x_fit_backward
            loss_backward.backward()

            # for p in model.parameters():
            #     p.grad.data.clamp_(-15, 15)
            
            optimizer.step()

            log_dict = {
                'loss_y_fit_forward' : loss_y_fit_forward.item(),
                'loss_yz_MMD_forward' : loss_yz_MMD_forward.item(),
                'loss_forward' : loss_forward.item(),
                'loss_x_MMD_backward' : loss_x_MMD_backward.item(),
                'loss_x_fit_backward' : loss_x_fit_backward.item(),
                'loss_backward' : loss_backward.item(),
            }
            total_steps += 1
            log.add(total_steps, log_dict, 'Train')
            print('Epoch %d | Batch %d:' % (e, batch_idx) + ''.join([' %s=%.4f' % (tag, val) for tag, val in log_dict.items()]))


        # Test
        with torch.no_grad():
            rev_x = model(pad_testset_onehot, True)[:, :x_ndim].cpu().numpy()
            pred_onehots = model(torch.cat([testset_x, padding_func(testset_x, pad_ndim - x_ndim, 0)], -1))[:, -y_ndim:]
            pred_labels = onehot_to_label(pred_onehots).cpu().numpy()

            pred_res = pred_labels == testset_y.cpu().numpy()
            pred_accuracy = np.sum(pred_res) / pred_res.shape[0]
            log.add(e, {'Pred Accuracy' : pred_accuracy}, 'Eval')
            print('Epoch %d: Pred Accuracy = %.4f' % (e, pred_accuracy))

        visualize_GMM_data(testset_x.cpu().numpy(), pred_labels, num_classes, log_dir + 'Forward_epoch%d.png' % e)
        visualize_GMM_data(rev_x, testset_y.cpu().numpy(), num_classes, log_dir + 'Backward_epoch%d.png' % e)

        if e >= epochs:
            break

    log.close()
    torch.save({'model' : model.state_dict(), 'optimizer' : optimizer.state_dict()}, log_dir + 'INN.pkl')


if __name__ == '__main__':
    main(epochs = 80)