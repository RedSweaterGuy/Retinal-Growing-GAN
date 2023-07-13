import numpy as np
import os
from scipy.ndimage import rotate
from PIL import Image
from utils import image_shape, DRIVE_files, STARE_files, random_perturbation
import cv2
import multiprocessing as mp
from functools import partial


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


def gen_one_image(index, img_files, vessel_files, mask_files, img_size, img_dir, vessel_dir, mask_dir, step_size, img_set):
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

    set_vessel = np.asarray(Image.open(vessel_filename)).astype(int)
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
    if img_set == "training": #only augment training data
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


def gen_data(run: dict):
    dataset = run['dataset']
    step_size = run['data_rotation']
    img_out_dir = os.path.join("data", "{}_{}".format(dataset, step_size))


    src_dirs = ["training", "test"]
    #src_dirs.append("data/{}/training/".format(dataset))
    #src_dirs.append("data/{}/test/".format(dataset))

    if os.path.isdir(img_out_dir):
        print(f"data for dataset {dataset} with rotation {step_size} already exists")
        return img_out_dir

    print(f"start generating data for dataset {dataset} with rotation {step_size}")
    os.makedirs(img_out_dir)


    for img_set in src_dirs:
        print(f"generating data for {img_set}")
        src_dir = f"data/{dataset}/{img_set}/"

        img_dir_base = os.path.join(img_out_dir, img_set, "images")
        vessel_dir_base = os.path.join(img_out_dir, img_set, "1st_manual")
        mask_dir_base = os.path.join(img_out_dir, img_set, "mask")

        img_dir = os.path.join(img_dir_base, dataset)
        vessel_dir = os.path.join(vessel_dir_base, dataset)
        mask_dir = os.path.join(mask_dir_base, dataset)

        os.makedirs(img_dir)
        os.makedirs(vessel_dir)
        os.makedirs(mask_dir)

        if dataset == 'DRIVE' or dataset == 'DEMO':
            img_files, vessel_files, mask_files = DRIVE_files(src_dir)
        elif dataset == 'STARE':
            img_files, vessel_files, mask_files = STARE_files(src_dir)

        img_size = (640, 640) if dataset == 'DRIVE' else (720, 720)
        
        pool = mp.Pool(mp.cpu_count())
        
        partial_process_item = partial(gen_one_image, img_files=img_files, vessel_files=vessel_files, mask_files=mask_files, img_size=img_size, img_dir=img_dir, vessel_dir=vessel_dir, mask_dir=mask_dir, step_size=step_size, img_set=img_set)
        pool.map(partial_process_item, range(len(img_files)))

        pool.close()
        pool.join()
        

    return img_out_dir
