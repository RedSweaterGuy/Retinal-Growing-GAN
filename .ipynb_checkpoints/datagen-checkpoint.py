import numpy as np
import os
from scipy.ndimage import rotate
from PIL import Image
from utils import image_shape, DRIVE_files, STARE_files, random_perturbation
import cv2


# M: Can't get rid of Image.open because cv2.imread does not support .gif images,
# which our manual segmentations are saved as.
def pad_imgs(imgs, img_size):
    img_h, img_w = imgs.shape[0], imgs.shape[1]
    target_h, target_w = img_size[0], img_size[1]
    if len(imgs.shape) == 3:
        d = imgs.shape[2]
        padded = np.zeros((target_h, target_w, d))
    elif len(imgs.shape) == 2:
        padded = np.zeros((target_h, target_w))

    padded[(target_h - img_h) // 2:(target_h - img_h) // 2 + img_h,
    (target_w - img_w) // 2:(target_w - img_w) // 2 + img_w, ...] = imgs

    return padded


def gen_data2(dataset, step_size):
    img_out_dir = os.path.join("data", "{}_{}".format(dataset, step_size))
    img_dir_base = os.path.join(img_out_dir, "images")
    vessel_dir_base = os.path.join(img_out_dir, "1st_manual")
    mask_dir_base = os.path.join(img_out_dir, "mask")
    img_dir = os.path.join(img_dir_base, dataset)
    vessel_dir = os.path.join(vessel_dir_base, dataset)
    mask_dir = os.path.join(mask_dir_base, dataset)
    src_dir = "data/{}/training/".format(dataset)

    if os.path.isdir(img_out_dir):
        return img_dir_base, vessel_dir_base, mask_dir_base

    os.makedirs(img_out_dir)
    os.makedirs(img_dir)
    os.makedirs(vessel_dir)
    os.makedirs(mask_dir)

    if dataset == 'DRIVE' or dataset == 'DEMO':
        img_files, vessel_files, mask_files = DRIVE_files(src_dir)
    elif dataset == 'STARE':
        img_files, vessel_files, mask_files = STARE_files(src_dir)

    img_size = (640, 640) if dataset == 'DRIVE' else (720, 720)

    for index in range(len(img_files)):
        img_filename = img_files[index]
        vessel_filename = vessel_files[index]
        mask_filename = mask_files[index]

        img_shape = image_shape(img_filename)
        vessel_shape = image_shape(vessel_filename)
        mask_shape = image_shape(mask_filename)

        # not needed
        set_imgs = np.zeros((1, img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
        set_vessels = np.zeros((1, vessel_shape[0], vessel_shape[1]), dtype=int)
        set_masks = np.zeros((1, mask_shape[0], mask_shape[1]), dtype=int)

        set_img = Image.open(img_filename)
        set_imgs[0] = np.asarray(set_img).astype(np.float32)
        set_imgs = pad_imgs(set_imgs, img_size)

        set_vessel = Image.open(vessel_filename)
        set_vessels[0] = np.asarray(set_vessel).astype(np.float32)
        set_vessels = pad_imgs(set_vessels, img_size)

        set_mask = Image.open(mask_filename)
        set_masks[0] = np.asarray(set_mask).astype(int)
        set_masks = pad_imgs(set_masks, img_size)

        # augment the original image (flip, rotate)
        all_set_imgs = [set_imgs]
        all_set_vessels = [set_vessels]

        flipped_imgs = set_imgs[:, :, ::-1, :]
        flipped_vessels = set_vessels[:, :, ::-1]

        all_set_imgs.append(flipped_imgs)
        all_set_vessels.append(flipped_vessels)

        for angle in range(step_size, 360, step_size):
            fig = rotate(set_vessels, angle, axes=(1, 2), reshape=False)
            print(f"fig shape: {fig.shape}")
            all_set_vessels.append(fig)
            all_set_vessels.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False))
            all_set_imgs.append(random_perturbation(rotate(set_imgs, angle, axes=(1, 2), reshape=False)))
            all_set_imgs.append(
                random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False)))

        set_vessels = np.asarray(np.round((np.concatenate(all_set_vessels, axis=0))), dtype=int)
        set_imgs = np.concatenate(all_set_imgs, axis=0, dtype=np.float32)

        # write files
        for j in range(len(set_imgs)):
            Image.fromarray((set_imgs[j, :, :]).astype(np.uint8)).save(
                os.path.join(img_dir,
                             "{:05}_{:03}.png".format(index + 1, j)))
            Image.fromarray((set_vessels[j, :, :]).astype(np.uint8)).save(
                os.path.join(vessel_dir,
                             "{:05}_{:03}.png".format(index + 1, j)))
        for j in range(len(set_masks)):
            Image.fromarray((set_masks[j, :, :]).astype(np.uint8)).save(
                os.path.join(mask_dir,
                             "{:05}_{:03}.png".format(index + 1, j)))

    return img_dir_base, vessel_dir_base, mask_dir_base


def gen_data(dataset, step_size):
    img_out_dir = os.path.join("data", "{}_{}".format(dataset, step_size))
    img_dir_base = os.path.join(img_out_dir, "images")
    vessel_dir_base = os.path.join(img_out_dir, "1st_manual")
    mask_dir_base = os.path.join(img_out_dir, "mask")
    img_dir = os.path.join(img_dir_base, dataset)
    vessel_dir = os.path.join(vessel_dir_base, dataset)
    mask_dir = os.path.join(mask_dir_base, dataset)
    src_dir = "data/{}/training/".format(dataset)

    if os.path.isdir(img_out_dir):
        return img_dir_base, vessel_dir_base, mask_dir_base

    os.makedirs(img_out_dir)
    os.makedirs(img_dir)
    os.makedirs(vessel_dir)
    os.makedirs(mask_dir)

    if dataset == 'DRIVE' or dataset == 'DEMO':
        img_files, vessel_files, mask_files = DRIVE_files(src_dir)
    elif dataset == 'STARE':
        img_files, vessel_files, mask_files = STARE_files(src_dir)

    img_size = (640, 640) if dataset == 'DRIVE' else (720, 720)

    for index in range(len(img_files)):
        img_filename = img_files[index]
        vessel_filename = vessel_files[index]
        mask_filename = mask_files[index]
        count = 0

        set_img = np.asarray(Image.open(img_filename)).astype(np.float32)
        set_img = pad_imgs(set_img, img_size)
        cv2.imwrite(
            os.path.join(img_dir, "{:05}_{:03}.png".format(index + 1, count)),
            cv2.cvtColor(np.float32(set_img), cv2.COLOR_RGB2BGR)
        )

        set_vessel = np.asarray(Image.open(vessel_filename)).astype(np.float32)
        set_vessel = pad_imgs(set_vessel, img_size)
        cv2.imwrite(
            os.path.join(vessel_dir, "{:05}_{:03}.png".format(index + 1, count)),
            set_vessel
        )

        set_mask = np.asarray(Image.open(mask_filename)).astype(int)
        set_mask = pad_imgs(set_mask, img_size)
        cv2.imwrite(
            os.path.join(mask_dir, "{:05}.png".format(index + 1)),
            set_mask
        )

        count += 1

        # augment the original image (flip, rotate)

        flipped_img = set_img[:, ::-1, :]
        flipped_vessel = set_vessel[:, ::-1]
        cv2.imwrite(
            os.path.join(img_dir, "{:05}_{:03}.png".format(index + 1, count)),
            cv2.cvtColor(np.float32(flipped_img), cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(vessel_dir, "{:05}_{:03}.png".format(index + 1, count)),
            flipped_vessel
        )
        count += 1
        for angle in range(step_size, 360, step_size):
            cv2.imwrite(
                os.path.join(vessel_dir, "{:05}_{:03}.png".format(index + 1, count)),
                rotate(set_vessel, angle, axes=(0, 1), reshape=False)
            )
            cv2.imwrite(
                os.path.join(vessel_dir, "{:05}_{:03}.png".format(index + 1, count + 1)),
                rotate(flipped_vessel, angle, axes=(0, 1), reshape=False)
            )
            cv2.imwrite(
                os.path.join(img_dir, "{:05}_{:03}.png".format(index + 1, count)),
                cv2.cvtColor(np.float32(random_perturbation(rotate(set_img, angle, axes=(0, 1), reshape=False))),
                             cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(img_dir, "{:05}_{:03}.png".format(index + 1, count + 1)),
                cv2.cvtColor(np.float32(random_perturbation(rotate(flipped_img, angle, axes=(0, 1), reshape=False))),
                             cv2.COLOR_RGB2BGR)
            )
            count += 2

    return img_dir_base, vessel_dir_base, mask_dir_base
