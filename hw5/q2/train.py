import argparse
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN
from dataset import CIFAR10Dataset
from generals import get_path

# TRAINING_SET = './train_cifar10_100.csv'
# TRAINING_SET = './train_cifar10_1000.csv'
TRAINING_SET = './train_cifar10_full.csv'

# TEST_SET = './test_cifar10_1000.csv'
TEST_SET = './test_cifar10_full.csv'

def train(args, model, device):
    epochs, optimizer, lr, batch_size, train_data, test_data, num_workers \
        = args.epochs, args.optimizer, args.lr, args.batch_size, args.train_data, args.test_data, args.num_workers
    
    train_path = TRAINING_SET if train_data is None else train_data
    test_path = TEST_SET if test_data is None else test_data

    train_set = CIFAR10Dataset(train_path, training=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    scaler = torch.cuda.amp.GradScaler()

    save_dir = get_path(os.path.join(os.getcwd(), 'runs', args.model_name))
    save_dir.mkdir(parents=True, exist_ok=True)
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'

    result = save_dir / 'result.txt'

    nb = len(train_loader)

    best_val_acc = 0

    for epoch in range(epochs):

        pbar = tqdm(enumerate(train_loader), total=nb)
        
        for i, (images, labels) in pbar:
            optim.zero_grad()

            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            pbar.set_description(f'[Training] Loss: {loss:.5f}')
        
        scheduler.step()

        val_acc, val_loss = val(args, model, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            print('*'*20)
            print(f'[New Best Accuracy] {val_acc*100:.5f}% on epoch: {epoch}')

            ckpt = {
                'epoch': epoch,
                'accuracy': val_acc,
                'model': model,
                'optimizer': optim.state_dict(),
                'opt': args,
                'date': datetime.now().isoformat()
            }

            torch.save(ckpt, best)
    
    print(f'Train ended on {epoch}')
    ckpt = {
        'epoch': epoch,
        'accuracy': val_acc,
        'model': model,
        'optimizer': optim.state_dict(),
        'opt': args,
        'date': datetime.now().isoformat()
    }

    torch.save(ckpt, last)

    result_txt = f'''
    Model Name: {args.model_name}
    Epochs: {epochs}
    Final loss: {val_loss}
    Best validation accuracy: {best_val_acc}    
    '''
    with open(result, 'w') as f:
        f.write(result_txt)




def val(args, model, device):
    epochs, optimizer, lr, batch_size, train_data, test_data, num_workers \
        = args.epochs, args.optimizer, args.lr, args.batch_size, args.train_data, args.test_data, args.num_workers

    test_path = TEST_SET if test_data is None else test_data

    test_set = CIFAR10Dataset(test_path, training=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    n = len(test_set)    
    nb = len(test_loader)

    pbar = tqdm(enumerate(test_loader), total=nb)

    loss = 0
    acc = 0
    
    for epoch, (images, labels) in pbar:
        
        b = len(images)

        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)
        
        batch_loss = criterion(preds, labels)
        batch_acc = torch.sum(torch.argmax(preds, dim=-1)==torch.argmax(labels, dim=-1)) / b
        loss += criterion(preds, labels) * b
        acc += batch_acc * b

        pbar.set_description(f'[Validation] Acc: {batch_acc*100:.5f}%, Loss: {batch_loss:.5f}')
    
    loss /= n
    acc /= n

    print(f'[Validation] total accuracy: {acc*100:.5f}%, total loss: {loss:.5f}')

    del test_set, test_loader

    return acc.cpu().detach().numpy(), loss.cpu().detach().numpy()



def main(opt):
    
    device = torch.device(opt.device)
    model = CNN().to(device)

    train(opt, model, device)
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--train-data', type=str, default='./train_cifar10_full.csv', help='ltrain data')
    parser.add_argument('--test-data', type=str, default='./test_cifar10_full.csv', help='test data')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, cuda:0 or cpu')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers')
    parser.add_argument('--model-name', type=str, default='CNN_big', help='name to save model')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)