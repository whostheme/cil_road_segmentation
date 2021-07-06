
import os
import torch
from torch import nn
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

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



from model import UNet
from train import train

def concat_vh(list_2d):
    # return final image
    return cv2.vconcat([cv2.hconcat(list_h)
                        for list_h in list_2d])


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
    train_dataset = ImageDataset(config['train'], device, use_patches=False, resize_to=(400, 400), test_run=test_run)  # resize to 384

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
    test_images = np.stack([cv2.resize(img, dsize=(608, 608)) for img in test_images], 0) # resize to 384
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    #test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = []
    for i, t in enumerate(test_images.unsqueeze(1)):
        print(i)

        top_left = t[:, :, :400, :400]
        top_right = t[:, :, :400, 208:608]
        bottom_left = t[:, :, 208:608, :400]
        bottom_right = t[:, :, 208:608, 208:608]
        top_left = model(top_left).detach().cpu().numpy().squeeze()
        top_right = model(top_right).detach().cpu().numpy().squeeze()
        bottom_left = model(bottom_left).detach().cpu().numpy().squeeze()
        bottom_right = model(bottom_right).detach().cpu().numpy().squeeze()

        img_tile = concat_vh([[top_left, top_right[:, 192:]],
                                [bottom_left[192:, :], bottom_right[192:, 192:]]
                                ])

        img_tile = np.expand_dims(np.expand_dims(img_tile, 0), 0)
        test_pred.append(img_tile)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(test_images[i][0].cpu(), cmap='gray')
        ax2.imshow(test_pred[i].squeeze(), cmap='gnuplot')
        plt.show()

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
