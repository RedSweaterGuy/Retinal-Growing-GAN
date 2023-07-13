import numpy as np

import model
# from model import GAN, discriminator_image, generator, pretrain_g
import utils
import os
from PIL import Image
import argparse
from keras import backend as K
import torch
import torchvision.datasets as dset
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from sklearn.model_selection import train_test_split


def print_time(text):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\n" + text + "\nCurrent Time =", current_time, "\n")


# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_dataset(dataroot, image_size, batch_size, workers, normalize=True):
    # Create the dataset
    dataset_pre_norm = dset.ImageFolder(root=dataroot,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                        ]))
    if not normalize:
        return dataset_pre_norm
    # Create the dataloader
    dataloader_pre_norm = torch.utils.data.DataLoader(dataset_pre_norm, batch_size=batch_size,
                                                      shuffle=False, num_workers=workers)
    mean, std = get_mean_and_std(dataloader_pre_norm)

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std),
                               ]))

    return dataset

class TrainDataset(Dataset):
    def __init__(self, image_data, vessel_data):
        self.images = image_data
        self.vessels = vessel_data
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        vessel = self.vessels[index]
        image = image[0].permute(1,2,0)
        vessel = vessel[0].permute(1,2,0)
        return image, vessel

class TestDataset(Dataset):
    def __init__(self, image_data, vessel_data, mask_data):
        self.images = image_data
        self.vessels = vessel_data
        self.masks = mask_data
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        vessel = self.vessels[index]
        mask = self.masks[index]
        image = image[0].permute(1, 2, 0)
        vessel = vessel[0].permute(1, 2, 0)
        mask = mask[0].permute(1, 2, 0)
        return image, vessel, mask
def get_train_dataset(dataroot, image_size, batch_size, workers, test_size):
    image_dataset = get_dataset(os.path.join(dataroot, "images"), image_size, batch_size, workers)
    vessel_dataset = get_dataset(os.path.join(dataroot, "1st_manual"), image_size, batch_size, workers, normalize=False)
    dataset = TrainDataset(image_dataset, vessel_dataset)
    image_train, image_val, vessel_train, vessel_val = train_test_split(dataset.images, dataset.vessels, test_size=test_size)

    train_dataloader = DataLoader(TrainDataset(image_train, vessel_train), batch_size=batch_size, num_workers=workers, shuffle=True)
    validation_dataloader = DataLoader(TrainDataset(image_val, vessel_val), batch_size=batch_size, num_workers=workers, shuffle=True)

    return train_dataloader, validation_dataloader


def get_test_dataset(dataroot, image_size, batch_size, workers):
    image_dataset = get_dataset(os.path.join(dataroot, "images"), image_size, batch_size, workers)
    vessel_dataset = get_dataset(os.path.join(dataroot, "1st_manual"), image_size, batch_size, workers, normalize=False)
    mask_dataset = get_dataset(os.path.join(dataroot, "mask"), image_size, batch_size, workers, normalize=False)
    dataset = TestDataset(image_dataset, vessel_dataset, mask_dataset)

    test_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                  shuffle=False)

    return test_dataloader


def train(run, img_src_dir):

    # training settings
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ratio_gan2seg = 1
    discriminator = "image"
    n_rounds = 10
    batch_size = 2
    n_filters_d = 32
    n_filters_g = 32
    val_ratio = 0.05
    init_lr = 2e-4
    # Number of workers for dataloader
    workers = 4
    schedules = {'lr_decay': {},  # learning rate and step have the same decay schedule (not necessarily the values)
                 'step_decay': {}}
    alpha_recip = 1. / ratio_gan2seg if ratio_gan2seg > 0 else 0
    rounds_for_evaluation = range(n_rounds)

    print_time("Start, loading data")

    # set dataset
    dataset = run['dataset']
    rotation = run['data_rotation']
    runs_dir = "runs"
    img_size = (640, 640) if dataset == 'DRIVE' else (720, 720)
    img_out_dir = os.path.join(runs_dir,"{}_{}/segmentation_results_{}_{}".format(dataset, rotation, discriminator, ratio_gan2seg))
    model_out_dir = os.path.join(runs_dir,"{}_{}/model_{}_{}".format(dataset,rotation, discriminator, ratio_gan2seg))
    auc_out_dir = os.path.join(runs_dir,"{}_{}/auc_{}_{}".format(dataset,rotation, discriminator, ratio_gan2seg))
    train_dir = os.path.join(img_src_dir, "training")
    test_dir = os.path.join(img_src_dir, "test")
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir)
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    # set training and validation dataset
    # dataloaders
    #all_imgs = get_dataset(train_dir, image_size=img_size[0], batch_size=batch_size,
    #                       workers=workers)
    #all_vessels = get_dataset(train_dir, image_size=img_size[0], batch_size=batch_size,
    #                          workers=workers)

    train_dataloader, val_dataloader = get_train_dataset(train_dir, image_size=img_size[0], batch_size=batch_size,
                           workers=workers, test_size=val_ratio)


    # train_vessels = np.expand_dims(train_vessels, axis=3)
    n_all_imgs = len(train_dataloader.dataset)+len(val_dataloader.dataset)
    n_train_imgs = int((1 - val_ratio) * n_all_imgs)
    #train_indices = np.random.choice(n_all_imgs, n_train_imgs, replace=False)
    #val_indices = np.delete(range(n_all_imgs), train_indices)
    #whole_dataset = torch.utils.data.ChainDataset([all_imgs, all_vessels])
    #train_dataloader = Subset(whole_dataset, train_indices)
    #val_dataloader = Subset(whole_dataset, val_indices)
    #whole_dataloader = torch.utils.data.DataLoader(whole_dataset, batch_size=batch_size, shuffle=True,
    #                                               num_workers=workers)

    # set test dataset
    test_dataloader = get_test_dataset(test_dir, image_size=img_size[0], batch_size=batch_size, workers=workers)
    #test_imgs = get_dataset(os.path.join(test_dir, "images"), image_size=img_size[0], batch_size=batch_size,
    #                        workers=workers)
    #test_vessels = get_dataset(os.path.join(test_dir, "1st_manual"), image_size=img_size[0], batch_size=batch_size,
    #                           workers=workers)
    #test_masks = get_dataset(os.path.join(test_dir, "mask"), image_size=img_size[0], batch_size=batch_size,
    #                         workers=workers)
    #test_dataset = torch.utils.data.ChainDataset([test_imgs, test_vessels, test_masks])
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
    #                                              num_workers=workers)

    # create networks
    g = model.generator(img_size, n_filters_g)

    d, d_out_shape = model.discriminator_image(img_size, n_filters_d, init_lr)

    utils.make_trainable(d, False)
    gan = model.GAN(g, d, img_size, n_filters_g, n_filters_d, alpha_recip, init_lr)
    g = model.pretrain_g(g, img_size, n_filters_g, init_lr) #changed generator to g because otherwise pretraining has no effect
    g.summary()
    d.summary()
    gan.summary()
    with open(os.path.join(model_out_dir, "g_{}_{}.json".format(discriminator, ratio_gan2seg)), 'w') as f:
        f.write(g.to_json())

    # start training
    scheduler = utils.Scheduler(n_train_imgs // batch_size, n_train_imgs // batch_size, schedules,
                                init_lr) if alpha_recip > 0 else utils.Scheduler(0, n_train_imgs // batch_size,
                                                                                 schedules, init_lr)
    print("training {} images :".format(n_train_imgs))
    for n_round in range(n_rounds):
        print_time("Start training round " + str(n_round + 1))

        # train D
        utils.make_trainable(d, True)
        for i, (real_imgs, real_vessels) in enumerate(train_dataloader, 0):
            real_imgs = real_imgs.numpy(force=True)
            real_vessels = np.expand_dims(real_vessels.numpy(force=True)[:,:,:,0], axis=3)
            d_x_batch, d_y_batch = utils.input2discriminator(real_imgs, real_vessels,
                                                             g.predict(real_imgs, batch_size=batch_size), d_out_shape)
            loss, acc = d.train_on_batch(d_x_batch, d_y_batch)

        # train G (freeze discriminator)
        utils.make_trainable(d, False)
        for i, (real_imgs, real_vessels) in enumerate(train_dataloader, 0):
            real_imgs = real_imgs.numpy(force=True)
            real_vessels = np.expand_dims(real_vessels.numpy(force=True)[:,:,:,0], axis=3)
            g_x_batch, g_y_batch = utils.input2gan(real_imgs, real_vessels, d_out_shape)
            loss, acc = gan.train_on_batch(g_x_batch, g_y_batch)

        """
        # evaluate on validation set
        if n_round in rounds_for_evaluation:
            for i, (val_imgs, val_vessels) in enumerate(val_dataloader, 0):
                val_imgs = val_imgs.numpy(force=True)
                val_vessels = np.expand_dims(val_vessels.numpy(force=True)[:,:,:,0], axis=3)
                # D
                d_x_test, d_y_test = utils.input2discriminator(val_imgs, val_vessels,
                                                               g.predict(val_imgs, batch_size=batch_size), d_out_shape)
                loss, acc = d.evaluate(d_x_test, d_y_test, batch_size=batch_size, verbose=0)
                utils.print_metrics(n_round + 1, loss=loss, acc=acc, type='D')
                # G
                gan_x_test, gan_y_test = utils.input2gan(val_imgs, val_vessels, d_out_shape)
                loss, acc = gan.evaluate(gan_x_test, gan_y_test, batch_size=batch_size, verbose=0)
                utils.print_metrics(n_round + 1, acc=acc, loss=loss, type='GAN')
        """
        # save the weights
        g.save_weights(os.path.join(model_out_dir,
                                    "g_{}_{}_{}.h5".format(n_round, discriminator, ratio_gan2seg)))

        # update step sizes, learning rates
        scheduler.update_steps(n_round)
        K.set_value(d.optimizer.lr, scheduler.get_lr())
        K.set_value(gan.optimizer.lr, scheduler.get_lr())

        # evaluate on test images
        if n_round in rounds_for_evaluation:
            for i, (test_imgs, test_vessels, test_masks) in enumerate(test_dataloader, 0):
                test_imgs = test_imgs.numpy(force=True)
                test_vessels = np.expand_dims(test_vessels.numpy(force=True)[:,:,:,0], axis=3)
                test_masks = test_masks.numpy(force=True)[:,:,:,0]
                generated = g.predict(test_imgs, batch_size=batch_size)
                generated = np.squeeze(generated, axis=3)
                vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels, generated, test_masks)
                auc_roc = utils.AUC_ROC(vessels_in_mask, generated_in_mask,
                                        os.path.join(auc_out_dir, "auc_roc_{}.npy".format(n_round)))
                auc_pr = utils.AUC_PR(vessels_in_mask, generated_in_mask,
                                      os.path.join(auc_out_dir, "auc_pr_{}.npy".format(n_round)))
                binarys_in_mask = utils.threshold_by_otsu(generated, test_masks)
                dice_coeff = utils.dice_coefficient_in_train(vessels_in_mask, binarys_in_mask)
                acc, sensitivity, specificity = utils.misc_measures(vessels_in_mask, binarys_in_mask)
                utils.print_metrics(n_round + 1, auc_pr=auc_pr, auc_roc=auc_roc, dice_coeff=dice_coeff,
                                    acc=acc, senstivity=sensitivity, specificity=specificity, type='TESTING')

                # print test images
                segmented_vessel = utils.remain_in_mask(generated, test_masks)
                for index in range(segmented_vessel.shape[0]):
                    Image.fromarray((segmented_vessel[index, :, :] * 255).astype(np.uint8)).save(
                        os.path.join(img_out_dir, str(n_round) + "_{:02}_segmented.png".format(index + 1)))
