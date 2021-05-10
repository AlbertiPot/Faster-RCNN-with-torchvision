import time
import torch
import torch.utils.data as d
import torchvision
import torchvision.transforms as transforms

import utils
import dataset.transforms as T # custom transforms
from dataset.coco_utils import get_coco, get_coco_kp
 
def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/home/gbc/workspace/coco/', get_coco, 91),
        "coco_kp": ('/datasets01/COCO/022719/', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name] # p应改为root

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

def main():
    # BATCH_SIZE = 100
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    # train_set = torchvision.datasets.MNIST('\mnist', download=False, train=True, transform=transform)
    
    # # data loaders
    # train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # distributed = 'False'
    batch_size = 8

    print("Loading data")
    dataset, num_classes = get_dataset('coco', "train", get_transform(train=True))

    train_sampler = torch.utils.data.RandomSampler(dataset)    
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    
    for num_workers in range(20):
        #train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, 
                                                num_workers=num_workers,
                                                collate_fn=utils.collate_fn)
        # training ...
        start = time.time()
        for epoch in range(1):
            #i = 0
            for images, targets in data_loader:
                #i = i+1
                #print(i)
                pass
        end = time.time()
        print('num_workers is {} and it took {} seconds'.format(num_workers, end - start))


if __name__ == '__main__':
    main()