'''
    Paper: Towards K-means friendly spaces: simultaneous deep learning and clustering
    Example implementation: https://github.com/xuyxu/Deep-Clustering-Network
'''


import torch
import torch.nn as nn
import numpy as np
import tqdm
import os
import itertools
import yaml
from sklearn.cluster import KMeans
from dataset import create_mnist_dataset, visualize_mnist_data
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils import Logger, create_mlp
from scipy.optimize import linear_sum_assignment


'''
    Neural network
'''
class MNISTFeatureExtractor(nn.Module):
    def __init__(self, hidden_dims = [500, 500, 2000], feature_dim = 10) -> None:
        super().__init__()

        # self.encoder = create_mlp(784, feature_dim, hidden_dims, act_fn = nn.ReLU)
        # self.decoder = create_mlp(feature_dim, 784, list(reversed(hidden_dims)), act_fn = nn.ReLU)

        self.encoder = []
        encoder_dims = [784] + hidden_dims
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            self.encoder.append(nn.Linear(in_dim, out_dim))
            self.encoder.append(nn.ReLU())
            # self.encoder.append(nn.BatchNorm1d(out_dim))
        self.encoder.append(nn.Linear(encoder_dims[-1], feature_dim))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        decoder_dims = [feature_dim] + list(reversed(hidden_dims))
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            self.decoder.append(nn.Linear(in_dim, out_dim))
            self.decoder.append(nn.ReLU())
            # self.decoder.append(nn.BatchNorm1d(out_dim))
        self.decoder.append(nn.Linear(decoder_dims[-1], 784))
        self.decoder = nn.Sequential(*self.decoder)


    def forward(self, x):
        feature = self.extract_feature(x)
        recon_x = self.decoder(feature)
        return feature, recon_x


    def extract_feature(self, x):
        return self.encoder(x.view(-1, 784))



class MNISTCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim = 10) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Flatten()
        )


    def extract_feature(self, x):
        return self.encoder(x)


    def forward(self, x):
        feature = self.extract_feature(x)
        z = self.decoder1(feature)
        z = z.view(-1, 256, 4, 4)
        z = self.decoder2(z)
        return feature, z




'''
    Clustering
'''
class MyKmeans(object):
    def __init__(self, feature_dim, n_clusters) -> None:
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.M = np.zeros((n_clusters, feature_dim))
        self.cluster_counter = np.ones((n_clusters)) * 100


    def init_cluster(self, features):
        kmeans = KMeans(self.n_clusters, n_init = 20)
        kmeans = kmeans.fit(features)
        self.M = kmeans.cluster_centers_


    def update_M(self, features, clusters):
        for f, c in zip(features, clusters):
            self.cluster_counter[c] += 1
            eta = 1.0 / self.cluster_counter[c]
            self.M[c] = (1 - eta) * self.M[c] + eta * f


    def assign_cluster(self, features):
        return np.argmin(np.sqrt(np.sum((features[:, None] - self.M[None])**2, axis = -1)), axis = -1)


    def save(self, path):
        np.savez(path, M = self.M, cluster_counter = self.cluster_counter)


    def load(self, path):
        with np.load(path) as f:
            self.M = f['M']
            self.cluster_counter = f['cluster_counter']


'''
    ACC
'''
def acc_func(y, ypred):
    """
    https://github.com/sarsbug/DCN_keras/blob/master/metrics.py
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    
    """
    assert len(y) > 0
    
    s = np.unique(ypred)
    t = np.unique(y)
    
    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype = np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    
    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    row,col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]] )
        count += np.count_nonzero(idx)
    
    return 1.0*count/len(y)




'''
    Main function
'''
def main(
    feature_dim = 10,
    hidden_dims = [500, 500, 2000],
    pretrain_epoch = 5,
    epoch = 20, 
    lr = 1e-3,
    lamda = 0.1,
    batch_size = 512, 
    log_path = './towards_kmeans_friendly_log/'
):
    # Dataset
    dataset = create_mnist_dataset(True)

    # DNN
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = MNISTFeatureExtractor(hidden_dims, feature_dim).to(device)
    model = MNISTCNNFeatureExtractor(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)

    # Kmeans 
    kmeans = MyKmeans(feature_dim, 10)

    # Log
    log = Logger(log_path)

    # Pre-training
    print('Start pre-training.')
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, True)
    model.train()
    for e in itertools.count(1):
        bar = tqdm.tqdm(dataset_loader)
        for img, _ in bar:
            img = img.to(device)
            _, recon_img = model(img)
            recon_loss = (img.view(-1, 784) - recon_img).pow(2).sum()
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            bar.set_description(f'Epoch {e}/{pretrain_epoch} - Recon_loss: {recon_loss.item():.6f}')

        if e >= pretrain_epoch:
            break


    # Initialize kmeans
    print('Initialize kmeans.')
    model.eval()
    with torch.no_grad():
        features = torch.cat([model.extract_feature(img.to(device)) for img, _ in torch.utils.data.DataLoader(dataset, batch_size, False)]).cpu().numpy()
    kmeans.init_cluster(features)


    # Update both DNN and kmeans
    print('Start running DCN.')
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size, False)
    for e in itertools.count(1):
        model.train()
        bar = tqdm.tqdm(dataset_loader)
        for img, _ in bar:
            img = img.to(device)

            with torch.no_grad():
                features_nograd = model.extract_feature(img).cpu().numpy()
            cluster_indices = kmeans.assign_cluster(features_nograd)
            kmeans.update_M(features_nograd, cluster_indices)

            features, recon_img = model(img)
            # Reconstruction loss
            recon_loss = (img.view(-1, 784) - recon_img).pow(2).sum()
            # Cluster loss
            M_ts = torch.from_numpy(kmeans.M).to(device)
            M_ts = torch.index_select(M_ts, 0, torch.from_numpy(cluster_indices).to(device))
            cluster_loss = 0.5 * lamda * (features - M_ts).pow(2).sum()
            # Total loss
            total_loss = recon_loss + cluster_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            bar.set_description(f'Epoch {e}/{epoch} - Recon_loss: {recon_loss.item():.6f} | Cluster_loss: {cluster_loss.item():.6f} | Total_loss: {total_loss.item():.6f}')


        if e % 5 == 0:
            model.eval()
            # Evaluation
            y_gt, y_pred = [], []
            with torch.no_grad():
                for img, label in dataset_loader:
                    img = img.to(device)
                    features = model.extract_feature(img).cpu().numpy()
                    clusters = kmeans.assign_cluster(features)
                    y_pred.append(clusters)
                    y_gt.append(label.cpu().numpy())
            y_gt, y_pred = np.concatenate(y_gt), np.concatenate(y_pred)
            ari = adjusted_rand_score(y_gt, y_pred)
            nmi = normalized_mutual_info_score(y_gt, y_pred)
            acc = acc_func(y_gt, y_pred)
            print(f'Epoch {e}/{epoch} - ARI: {ari:.6f} | NMI: {nmi:.6f} | ACC: {acc:.6f}')
            log.add(e, {'ARI' : ari, 'NMI' : nmi, 'ACC' : acc}, 'Eval/')

            # Visualize the performance (Sometimes the empty cluster problem exists!)
            vis_imgs = []
            for i in range(10):
                test_indices = np.where(y_pred == i)[0][:16]
                if len(test_indices) < 16:
                    vis_imgs.append(torch.zeros((16, 1, 28, 28)))
                else:
                    cluster_vis_imgs = [dataset[idx][0] for idx in test_indices]
                    vis_imgs.append(torch.stack(cluster_vis_imgs))
            vis_imgs = torch.cat(vis_imgs)
            visualize_mnist_data(vis_imgs, save_path = log_path + f'eval_epoch{e}.png', n_row = 16)

        
        if e >= epoch:
            break


    # Save models
    state_dict = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    torch.save(state_dict, log_path + 'dnn.pkl')
    kmeans.save(log_path + 'kmeans.npz')


if __name__ == '__main__':
    config = dict(
        feature_dim = 10,
        hidden_dims = [500, 500, 2000],
        pretrain_epoch = 100,
        epoch = 100,
        lr = 1e-3,
        lamda = 1,
        batch_size = 512,
        log_path = './towards_kmeans_friendly_log13/'
    )
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
    main(**config)