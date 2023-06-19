import os
from imports import *
from utils import *
from models import *
from dataloader import *
from training_function import*

if __name__ == "__main__":
    data_path = "/content/drive/MyDrive"
    domain_a_path = os.path.join(data_path, "Pix2PixA_train")
    domain_b_path = os.path.join(data_path, "Pix2PixB_train2")

    transform = Compose([
        Resize((140, 250)),
        ToTensor(),
    ])

    paired_dataset = PairedDataset(domain_a_path, domain_b_path, transform=transform)
    num_workers = os.cpu_count()
    paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=False, num_workers=16)


    epochs = 200
    train_pix2pix(paired_loader, epochs)
