import torch
import torch.nn as nn
from utils import create_mlp, Logger
from dataset import create_GMM_dataset, visualize_GMM_data, onehot_to_label, label_to_onehot
from itertools import count


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, hidden_dim = 256) -> None:
        '''
            x: data point
            c: data label (conditioned?)
        '''
        super().__init__()

        self.encoder = create_mlp(x_dim + c_dim, 2 * z_dim, [hidden_dim], act_fn = nn.ReLU)
        self.decoder = create_mlp(z_dim + c_dim, x_dim, [hidden_dim], act_fn = nn.ReLU)


    def encode(self, x, c):
        z_out = self.encoder(torch.cat([x, c], -1))
        z_mu, z_logvar = torch.chunk(z_out, 2, -1)
        # z_logvar = torch.clamp(z_logvar, -10, 2)
        return z_mu, z_logvar


    def reparameterization(self, mu, logvar):
        return mu + torch.exp(logvar * 0.5) * torch.randn_like(logvar, requires_grad = True)


    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], -1))


def recon_loss(x, x_sample):
    # return torch.nn.functional.binary_cross_entropy(x_sample, x)
    return 0.5 * (x - x_sample).pow(2).mean()


def kl_loss(mu, logvar):
    return (0.5 * torch.sum(logvar.exp() + mu**2 - 1 - logvar, -1)).mean()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = './cvae_exp/'
    log = Logger(log_dir)

    # Generate dataset
    radius1 = 2.0
    radius2 = 0.15
    batch_size = 2048
    num_classes = 8
    trainset_loader = create_GMM_dataset(100000, num_classes, radius1, radius2, batch_size)
    testset_loader = create_GMM_dataset(4000, num_classes, radius1, radius2, 10000)
    testset_x, testset_c = next(iter(testset_loader))
    visualize_GMM_data(testset_x.numpy(), testset_c.numpy(), num_classes, log_dir + 'testset.png')
    testset_x, testset_c = testset_x.to(device), testset_c.to(device)

    # Create the network
    x_dim = 2
    c_dim = num_classes
    z_dim = 64
    hidden_dim = 256
    model = ConditionalVAE(x_dim, c_dim, z_dim, hidden_dim).to(device)
    # model.apply(torch.nn.init.xavier_normal_)
    optimzier = torch.optim.Adam(model.parameters(), lr = 1e-3)

    epochs = 30
    total_batches = 0 
    for e in count(1):
        # Train
        for batch_idx, (x, c) in enumerate(trainset_loader):
            total_batches += 1
            x, c = x.to(device), c.to(device)
            c = label_to_onehot(c, num_classes)

            optimzier.zero_grad()
            z_mu, z_logvar = model.encode(x, c)
            z = model.reparameterization(z_mu, z_logvar)
            x_sample = model.decode(z, c)

            rl = recon_loss(x, x_sample)
            kl = kl_loss(z_mu, z_logvar)
            loss = rl + kl
            loss.backward()
            optimzier.step()

            log_dict = {'recon_loss' : rl.item(), 'kl_loss' : kl.item(), 'total_loss' : loss.item()}
            log.add(total_batches, log_dict)
            print('Epoch %d | Batch %d:' % (e, batch_idx) + ''.join([' %s=%.4f' % (tag, val) for tag, val in log_dict.items()]))

        # Test
        with torch.no_grad():
            bs = testset_c.size(0)
            z = torch.randn(bs, z_dim).to(device)
            test_samples = model.decode(z, label_to_onehot(testset_c, num_classes)).cpu().numpy()
        
        visualize_GMM_data(test_samples, testset_c.cpu().numpy(), num_classes, log_dir + 'epoch%d_sample.png' % e)

        if e >= epochs:
            break

    log.close()
    torch.save(
        {'cvae' : model.state_dict(), 'optimizer' : optimzier.state_dict()}, 
        log_dir + 'cvae.pkl'
    )