from copy import deepcopy
import cupy as cp
from tqdm import tqdm
import pickle

from utils.dataset import DataLoader, Dataset 
import utils.optimizer as op
import utils.loss as ls
from utils.model import CNN, CNN2, CNN3
from utils.common import softmax


TRAINING_SET = './mnist_200.csv'
TEST_SET = './mnist_20+6.csv'

N_EPOCHS = 100
VAL_EPOCHS = 10
lr = 0.001


# model = CNN()
# model = CNN2()
model = CNN3()

model.explain()

def training():

    criterion = ls.CrossEntropyWithLogits()
    optimizer = op.SGD(model, criterion, lr=lr)
    # optimizer = op.Adam(model, criterion, lr=lr)

    train_set = Dataset(TRAINING_SET, 200)

    n = len(train_set)

    train_loader = DataLoader(train_set, training=True, batch_size=20)

    best_val_acc = 0
    best_model = None
    best_epoch = 0
    
    for epoch in range(N_EPOCHS):

        correct = 0
        total_loss = 0

        pbar = tqdm(enumerate(train_loader))

        for i, (images, labels) in pbar:

            n_batch = len(labels)

            optimizer.zero_grad()
            outputs = model.forward(images, learning=True)
            # print(output.shape)
            loss = criterion(outputs, labels)
            # print(loss_val.shape)
            loss_grad = criterion.backprop(outputs, labels)
            # print(loss_grad.shape)
            model.backward(loss_grad)
            optimizer.step()

            total_loss += loss * n_batch

            outputs = cp.argmax(outputs, axis=-1)
            labels = cp.argmax(labels, axis=-1)

            correct += cp.sum(outputs == labels)

            pbar.set_description(f'Training epoch: {epoch}, train loss={loss}')
        
        acc = correct / n
        total_loss /= n

        print(f'-'*15)
        print(f'Epoch {epoch} done, train acc={acc:.5f}, train loss={total_loss:.5f}')
        
        if epoch%VAL_EPOCHS==0:
            val_acc = eval()
            print(f'-'*15)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = deepcopy(model)
                best_epoch = epoch
        

    val_acc = eval()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = deepcopy(model)
        best_epoch = epoch
    
    print(f'*'*10)
    print(f'Training done.')
    print(f'Training acc: {acc}')
    print(f'Validatioin acc: {val_acc}')
    print(f'Best Validatioin acc: {best_val_acc} at epoch {best_epoch}')
    
    if best_model is not None:
        save_object(best_model, './best_model.pkl')
    save_object(model, './last_model.pkl')

def eval():

    loss = ls.CrossEntropyWithLogits()

    test_set = Dataset(TEST_SET, 26)
    test_loader = DataLoader(test_set, training=False, batch_size=26)

    n = len(test_set)
    correct = 0

    pbar = tqdm(enumerate(test_loader), desc='Testing')

    for i, (images, labels) in pbar:
        outputs = model.forward(images, learning=True)
        # print(output.shape)
        loss_val = loss(outputs, labels)
        # print(loss_val.shape)

        pbar.set_description(f'Testing loss = {loss_val}')

        outputs = cp.argmax(outputs, axis=-1)
        labels = cp.argmax(labels, axis=-1)

        correct += cp.sum(outputs == labels)
    
    acc = correct / n

    print(f'Validation accuracy = {acc:.7f}')

    return acc

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    training()

