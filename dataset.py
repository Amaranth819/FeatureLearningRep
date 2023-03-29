import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

'''
    Create MNIST dataset
'''
def create_mnist_dataset(is_train):
    '''
        Return:
            images of size [bs, 1, 28, 28]
            number labels of size [bs]
    '''
    return torchvision.datasets.MNIST(
        './datasets/MNIST/', train = is_train,
        download = True, transform = torchvision.transforms.ToTensor()
    )


def visualize_mnist_data(imgs, title = None, save_path = 'mnist.png', n_row = 8):
    '''
        imgs: [n_imgs, 1, 28, 28] tensors
        labels: [n_imgs] or None
    '''
    grid = make_grid(imgs, nrow = n_row)
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()



'''
    Create Gaussian mixture model dataset
'''
def label_to_onehot(label, num_classes = -1):
    return torch.eye(torch.max(label) + 1 if num_classes == -1 else num_classes)[label].to(label.device)


def onehot_to_label(oh):
    return torch.argmax(oh, 1)


def create_GMM_dataset(dataset_size, num_classes, radius1 = 2.0, radius2 = 0.15, batch_size = 128):
    num_points_per_class = dataset_size // num_classes
    data_points = []
    labels = []
    
    for i in range(num_classes):
        theta = i / num_classes * 2 * np.pi
        fixed_point = np.array([radius1 * np.cos(theta), radius1 * np.sin(theta)])[None]
        data_points.append(np.random.randn(num_points_per_class, 2) * radius2 + fixed_point)
        labels.append(np.ones(num_points_per_class, dtype = int) * i)

    data_points = np.concatenate(data_points) 
    labels = np.concatenate(labels) 

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(data_points).float(), torch.from_numpy(labels)),
        batch_size = batch_size,
        shuffle = True
    )

    return loader


def visualize_GMM_data(data_points, labels, num_classes = -1, save_path = 'vis.png'):
    '''
        data_points: [num_points, 2] (2 for xy)
        labels: [num_points]
    '''
    # Currently only support at most 8 labels
    color_set = ['red', 'blue', 'yellow', 'black', 'green', 'gray', 'pink', 'purple']

    if num_classes == -1:
        num_classes = np.max(labels)
    for label in np.arange(num_classes):
        point_indices = np.where(labels == label)[0]
        label_points = data_points[point_indices]
        plt.scatter(label_points[:, 0], label_points[:, 1], color = color_set[label], s = 0.5, label = 'class %d' % label)
    plt.legend(loc = 'best')
    # plt.show()
    plt.savefig(save_path)
    plt.close()



if __name__ == '__main__':
    # num_classes = 8
    # test = create_GMM_dataset(1000, num_classes)
    # # for dp, l in test:
    # #     print(dp.size(), l.size())

    # data_points, labels = next(iter(test))
    # visualize_GMM_data(data_points.cpu().numpy(), labels.cpu().numpy(), num_classes)


    mnist = create_mnist_dataset(False)
    img = mnist[0][0]
    print(img.max(), img.min())