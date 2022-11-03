import torch
from torch.utils.data import DataLoader

from model import CNN
from dataset import CIFAR10Dataset

# TRAINING_SET = './train_cifar10_100.csv'
# TRAINING_SET = './train_cifar10_1000.csv'
TRAINING_SET = './train_cifar10_full.csv'

# TEST_SET = './test_cifar10_1000.csv'
TEST_SET = './test_cifar10_full.csv'

def training(args, model):
    lr, batch_size, train_data, test_data \
        = args.lr, args.batch_size, args.train_data, args.test_data
    
    train_path = TRAINING_SET if train_data is None else train_data
    test_path = TEST_SET if test_data is None else test_data

    train_set = CIFAR10Dataset(TRAINING_SET, training=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.RAdam(model, lr=lr)

    pbar = enumerate(train_loader)
    

