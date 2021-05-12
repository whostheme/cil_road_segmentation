
import os
import torch
from torch import nn
from glob import glob
import numpy as np
import cv2

from torch.utils.data import DataLoader, random_split


from utils import (
    read_json,
    read_args,
    make_path_timestamped,
    np_to_tensor,
    show_val_samples,
    load_all_from_path, 
    accuracy_fn,
    patch_accuracy_fn,
    create_submission
)

from get_data import(
    ImageDataset
)

from loss import(
    get_loss
)

from model import UNet
from train import train


def main():
    # Get arguments
    args = read_args()
    config = read_json(args.config)
    test_run = args.test
    load_from_model = args.load

    # Can be used to store experiment
    work_path = make_path_timestamped(config['safe_model_path'], config['name'])
    config['model_save_path'] = work_path
    os.makedirs(work_path)

    # Some constants
    PATCH_SIZE = config['patch_size']  # pixels per side of square patches
    CUTOFF = config['cutoff']  # minimum average brightness for a mask patch to be classified as containing road
    batch_size = config['batch_size']
    
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Reshape the image to simplify the handling of skip connections and maxpooling    
    train_dataset = ImageDataset(config['train'], device, use_patches=False, resize_to=(384, 384), test_run=test_run)  # resize to 384

    # Split dataset in train/validation set
    
    n_validation = int(len(train_dataset) * config['validation_size'])
    n_train = len(train_dataset) - n_validation
    train_split, val_split = random_split(train_dataset, [n_train, n_validation])
   
    train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True)
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = config['epochs']
    if(test_run):
        n_epochs = 1

    # Can be loaded from saved model
    if(load_from_model):
        model = UNet().to(device)
        model.load_state_dict(torch.load('model.pt'))
        model.eval()

    # Train model  
    else:
        train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, CUTOFF, PATCH_SIZE)
        torch.save(model.state_dict(), 'model.pt')
    
    
    # Predict on test set
    test_filenames = (glob(config['test_images'] + '/*.png'))
    test_images = load_all_from_path(config['test_images'])
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0) # resize to 384
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv', PATCH_SIZE=PATCH_SIZE)
        


if __name__ == "__main__":
    main()
