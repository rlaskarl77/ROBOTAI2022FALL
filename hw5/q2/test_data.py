import numpy as np

TRAINING_SET = './train_cifar10_full.csv'

with open(TRAINING_SET, 'r') as f:
    datas = f.readlines()
    n = len(datas)
    labels = np.zeros((10,))
    images = np.zeros((n, 32, 32))
    for i, data in enumerate(datas):
        values = data.strip().split(',')
        # print(values)
        label = int(values[0])
        try:
            image = np.array(list(map(float, values[1:1025]))).reshape(32, 32)/255.0
        except ValueError:
            print(i)
            print(label)
            # print(values)
            print(len(values))
            continue
        labels[label] += 1
        images[i] = image

print('total labels:', np.sum(labels))
print(labels)
print(labels/np.sum(labels))

print(np.mean(images))
print(np.var(images))
print(np.std(images))

