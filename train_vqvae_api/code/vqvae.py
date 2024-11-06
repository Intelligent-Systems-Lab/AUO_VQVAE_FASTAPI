import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import json
from modules import VectorQuantizedVAE, to_scalar
import cv2
import yaml
from dataset import C101_Dataset
from argparse import Namespace
import argparse
import os
import multiprocessing as mp
import shutil
import random
from tensorboardX import SummaryWriter

def train(data_loader, model, optimizer, args):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        optimizer.step()
        args.steps += 1

def test(data_loader, model, args):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde
def dict_to_namespace(d):
    return Namespace(**d)
def save_checkpoint(state, j_id=None):
    filename='best.pt'
    try:
        if j_id is None:
            raise ValueError("j_id is None")  
        
        job_path = os.path.join("./jobs", str(j_id))

        # 将保存的文件路径修改为新创建的目录下
        save_path = os.path.join(job_path, filename)
        
        # 保存 checkpoint 文件到新的路径
        torch.save(state, save_path)
 
        print(f"Checkpoint saved as {save_path}")
        modelcfg_path = os.path.join("./configs", "config.yaml")
        shutil.copy2(modelcfg_path, job_path)

            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main(args,j_id):
    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    # save_filename = './models/{0}'.format(args.output_folder)

    transform = transforms.Compose([
        transforms.Resize((256, 256),interpolation=cv2.INTER_NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    # Define the train, valid & test datasets
    # 設定分割比例
    train_ratio = 0.8

    # 目標資料夾
    train_dir = os.path.join(args.data_dir,'train')
    val_dir = os.path.join(args.data_dir,'valid')
    # 如果目標資料夾不存在，則建立
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 獲取資料夾中的所有圖片檔案
    all_files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]

    # 隨機打亂檔案順序
    random.shuffle(all_files)

    # 計算分割點
    split_index = int(len(all_files) * train_ratio)

    # 分割成 train 和 validation 資料
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # 將圖片移動到 train 資料夾
    for file in train_files:
        shutil.move(os.path.join(args.data_dir, file), os.path.join(train_dir, file))

    # 將圖片移動到 validation 資料夾
    for file in val_files:
        shutil.move(os.path.join(args.data_dir, file), os.path.join(val_dir, file))
    train_dataset = C101_Dataset(folder_path=train_dir, transform=transform)
    valid_dataset = C101_Dataset(folder_path=val_dir, transform=transform)

    num_channels = 1

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args)
        loss, _ = test(valid_loader, model, args)

        print("epoch: "+str(epoch)+", loss: "+str(loss))

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            save_checkpoint(model.state_dict(),j_id)
        #     with open('{0}/best.pt'.format(save_filename), 'wb') as f:
        #         torch.save(model.state_dict(), f)
        # with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
        #     torch.save(model.state_dict(), f)

    # for watchdog
    status = dict()
    status['exp_name'] = args.exp_name
    status['epoch'] = epoch
    status['best_loss'] = best_loss
    status['status'] = "Training"
    status['idle'] = True
    status['completed'] = True
    with open('./status.json', 'w') as f:
        json.dump(status, f)

if __name__ == '__main__':


    # 讀取 YAML 配置文件
    with open('./configs/config.yaml', 'r') as file:
        args = yaml.safe_load(file)
        args = dict_to_namespace(args)
        print(args)
    
    parser = argparse.ArgumentParser(description='VQ-VAE')

    # # General
    parser.add_argument('--j_id', type=int, required=True, help="Job ID for this training session")
    temp = parser.parse_args()

    j_id = temp.j_id

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    print(args.device)

    args.steps = 0

    # for watchdog
    status = dict()
    status['exp_name'] = args.exp_name
    status['epoch'] = 0
    status['best_loss'] = 100
    status['status'] = "Training"
    status['idle'] = False
    status['completed'] = False
    with open('./status.json', 'w') as f:
        json.dump(status, f)

    main(args,j_id)
