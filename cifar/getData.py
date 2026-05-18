from IPython.testing import test
import torch
import torchvision
import numpy as np


trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

all_images, all_labels = next(iter(train_loader))
all_test_image, all_test_lable = next(iter(test_loader))

images_np = (all_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
labels_np = all_labels.numpy()

test_images_np = (all_test_image.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
test_labels_np = all_test_lable.numpy()

np.save('cifar100_train_images.npy', images_np)
np.save('cifar100_train_labels.npy', labels_np)

np.save('cifar100_test_images.npy', test_images_np)
np.save('cifar100_test_labels.npy', test_labels_np)

print(f"Saved images: {images_np.shape} to cifar100_train_images.npy")
print(f"Saved labels: {labels_np.shape} to cifar100_train_labels.npy")
print(f"Saved test images: {test_images_np.shape} to cifar100_test_images.npy")
print(f"Saved test labels: {test_labels_np.shape} to cifar100_test_labels.npy")