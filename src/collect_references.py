import os
from torchvision import transforms
import torchvision
import torch 

dataset_path = "dataset"
save_path = "reference_images_128/faces"

num_images = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.RandomCrop((128, 128)),
])

dataset = torchvision.datasets.ImageFolder(
    root=dataset_path, 
    transform= transform
)
    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
os.makedirs(save_path, exist_ok=True)


k = 0
for img in dataloader:
    for i in range(img[0].shape[0]):
        torchvision.utils.save_image(
            img[0][i], 
            os.path.join(save_path, f"image_{k:03d}.png")
        )
        k += 1
        if k >= num_images:
            break
    if k >= num_images:
        break


