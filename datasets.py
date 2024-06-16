import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.distributions import MultivariateNormal
import hydra

import utils


class CustomTensorDataset(Dataset):
    def __init__(self, *tensors, transform=None) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        assert len(tensors) == 2, "tensors should have length 2"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def load_dataset(args):
    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
    train_len = args.data.train_len
    valid_len = args.data.valid_len
    test_len = args.data.test_len

    if args.dataset_name == "gaussian":
        assert len(args.obs_dim) == 1
        obs_dim = args.obs_dim[0]
        mu = torch.zeros(obs_dim)
        stds = torch.eye(obs_dim)
        distr = MultivariateNormal(mu, stds)

        train_set = TensorDataset(distr.sample((train_len,)), torch.zeros(train_len))
        valid_set = TensorDataset(distr.sample((valid_len,)), torch.zeros(valid_len))
        test_set = TensorDataset(distr.sample((test_len,)), torch.zeros(test_len))

    elif args.dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[args.dataset_name]
        local_data_dir = os.path.join(data_dir, "MNIST" if args.dataset_name == "mnist" else "FashionMNIST")

        train_set = dataset_class(data_dir, download=True, train=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(*args.obs_dim)), transforms.Lambda(torch.bernoulli)]))
        test_set = dataset_class(data_dir, download=True, train=False,
                                 transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(*args.obs_dim)), transforms.Lambda(torch.bernoulli)]))

        train_set, valid_set = random_split(train_set, [train_len, valid_len], generator=torch.Generator().manual_seed(42))
        
        assert len(test_set) == test_len

    elif args.dataset_name == "omniglot":
        import scipy.io
        import urllib
        local_data_dir = os.path.join(data_dir, "omniglot")
        raw_filename = os.path.join(local_data_dir, "chardata.mat")
        try:
            raw_data = scipy.io.loadmat(raw_filename)
        except FileNotFoundError:
            os.mkdir(local_data_dir)
            url = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat"
            print('Downloading from {}...'.format(url))
            urllib.request.urlretrieve(url, raw_filename)
            print('Saved to {}'.format(raw_filename))
            raw_data = scipy.io.loadmat(raw_filename)

        train_images = torch.from_numpy(np.transpose(raw_data["data"]))
        train_labels = torch.from_numpy(np.transpose(raw_data["targetchar"]))
        test_images = torch.from_numpy(np.transpose(raw_data["testdata"]))
        test_labels = torch.from_numpy(np.transpose(raw_data["testtargetchar"]))
        train_set = CustomTensorDataset(train_images, train_labels, 
                                        transform=transforms.Compose([transforms.Lambda(lambda x: x.view(*args.obs_dim)), transforms.Lambda(torch.bernoulli)]))
        test_set = CustomTensorDataset(test_images, test_labels, 
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x.view(*args.obs_dim)), transforms.Lambda(torch.bernoulli)]))

        train_set, valid_set = random_split(train_set, [train_len, valid_len], generator=torch.Generator().manual_seed(42))
        
        assert len(test_set) == test_len

    else:
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)

    if args.dataset_name in ["mnist", "fashion-mnist", "omniglot"]:
        utils.save_image(next(iter(DataLoader(train_set, batch_size=64)))[0], args, os.path.join(local_data_dir, 'im_grid_train.png'))
        utils.save_image(next(iter(DataLoader(valid_set, batch_size=64)))[0], args, os.path.join(local_data_dir, 'im_grid_valid.png'))
        utils.save_image(next(iter(DataLoader(test_set, batch_size=64)))[0], args, os.path.join(local_data_dir, 'im_grid_test.png'))

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    from omegaconf import OmegaConf
    # config = {'dataset_name': 'mnist', 'data': {'type': 'image', 'image_size': 28, 'channels': 1, 'train_len': 50000, 'valid_len': 10000, 'test_len': 10000}, 'paths': {'data_dir_name': 'data'}, 'batch_size': 1, 'test_batch_size': 1, 'obs_dim': [784]}
    # config = {'dataset_name': 'omniglot', 'data': {'type': 'image', 'image_size': 28, 'channels': 1, 'train_len': 23000, 'valid_len': 1345, 'test_len': 8070}, 'paths': {'data_dir_name': 'data'}, 'batch_size': 1, 'test_batch_size': 1, 'obs_dim': [784]}
    config = {'dataset_name': 'fashion-mnist', 'data': {'type': 'image', 'image_size': 28, 'channels': 1, 'train_len': 50000, 'valid_len': 10000, 'test_len': 10000}, 'paths': {'data_dir_name': 'data'}, 'batch_size': 1, 'test_batch_size': 1, 'obs_dim': [784]}
    args = OmegaConf.create(config)
    load_dataset(args)
