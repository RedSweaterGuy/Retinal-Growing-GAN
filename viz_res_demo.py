import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.transform import resize
from skimage.morphology import dilation
from matplotlib.colors import ListedColormap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

show_titles = True
rotate_by = 0
query_ground_truth = True
_cmap = "gray"


# maybe change this later
custom_colors = [
    "black",  # Black if image1 == image2 and image1 == 0
    "white",  # White if image1 == image2 and image1 == 1
    "orange",  # Red for image1 > image2
    "red"  # Green for image2 > image1
]
# Create the custom colormap
custom_cmap = ListedColormap(custom_colors)

dataset = "DRIVE"
methods = ['origGAN', 'growGAN', 'growGAN20']
titles = ['Original GAN', 'Growing GAN 10 rounds', 'Growing GAN 20 rounds']

img_size = (640, 640)
f_model = "pretrained/{}/{}_best.json"
f_weights = "pretrained/{}/{}_best.h5"
batch_size = 1


def remain_in_mask(img, masks):
    img[masks < 0.2] = 0
    return img


def multi_dilation(image, kernel, iterations):
    for i in range(iterations):
        image = dilation(image, kernel)
    return image


cross = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]])

# Flow:
# ask for image
# make segmentation with the three models
# output results with matplotlib
models = []
for i in range(len(methods)):
    # load the model and weights
    with open(f_model.format(dataset, methods[i]), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(f_weights.format(dataset, methods[i]))
    models.append(model)

# prevents an empty tkinter window from appearing
root = tk.Tk()
root.withdraw()

while True:
    segmentations = dict()  # dict where all segmentations for one run will be stored
    # first plot
    figs, axs = plt.subplots(2, len(methods) + (1 * query_ground_truth), sharex=True, sharey=True)
    for ax_s in axs:
        for ax in ax_s:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    image_path = filedialog.askopenfilename()

    img = Image.open(image_path)
    if rotate_by > 0:
        img = img.rotate(rotate_by)
    img = np.asarray(img).astype(np.float32)
    # set test dataset
    test_imgs = resize(img, img_size)[None, :, :, :]
    mean = np.mean(test_imgs[test_imgs[..., 0] > 40.0], axis=0, dtype=np.float32)
    std = np.std(test_imgs[test_imgs[..., 0] > 40.0], axis=0, dtype=np.float32)
    assert len(mean) == 3 and len(std) == 3
    orig_imgs = np.asarray(test_imgs) / 255
    orig_copy = np.copy(orig_imgs)
    test_imgs[0, ...] = (test_imgs[0, ...] - mean) / std

    ground_truth_path = image_path.replace("images", "1st_manual").replace("_test", "_manual1")
    if "DRIVE" in ground_truth_path:
        ground_truth_path = ground_truth_path.replace(".tif", ".gif")
    else:
        ground_truth_path = ground_truth_path.replace(".ppm", ".ah.ppm")
    ground_truth_img = Image.open(ground_truth_path)
    if rotate_by > 0:
        ground_truth_img = ground_truth_img.rotate(rotate_by)
    ground_truth_img = np.asarray(ground_truth_img).astype(np.float32)
    ground_truth_img = resize(ground_truth_img, img_size)
    segmentations['ground_truth_img'] = ground_truth_img
    axs[0, 0].imshow(ground_truth_img, cmap=_cmap)

    axs[1, 0].imshow(remain_in_mask(orig_imgs[0], multi_dilation(ground_truth_img, cross, 4)), cmap=_cmap)
    if show_titles:
        axs[0, 0].set_title("Ground truth")

    for i in range(len(methods)):
        generated = models[i].predict(test_imgs, batch_size=batch_size)
        generated = np.squeeze(generated, axis=3)[0]
        segmentations[methods[i]] = generated
        axs[0, i + (1 * query_ground_truth)].imshow(generated, cmap=_cmap)
        orig_imgs = np.copy(orig_copy)
        axs[1, i + (1 * query_ground_truth)].imshow(remain_in_mask(orig_imgs[0], multi_dilation(generated, cross, 4)),
                                                    cmap=_cmap)

        if show_titles:
            axs[0, i + (1 * query_ground_truth)].set_title(titles[i])
    plt.tight_layout()
    plt.show()

    # second plot
    figs, axs = plt.subplots(2, len(methods) + (1 * query_ground_truth), sharex=True, sharey=True)

    for ax_s in axs:
        for ax in ax_s:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

    if query_ground_truth:
        axs[0, 0].imshow(ground_truth_img, cmap=_cmap)

        axs[1, 0].imshow(remain_in_mask(orig_imgs[0], ground_truth_img), cmap=_cmap)
        if show_titles:
            axs[0, 0].set_title("Ground truth")
    for i in range(len(methods)):
        fake = np.where(segmentations[methods[i]].copy() > 0.1,1,0)
        grd = np.where(segmentations['ground_truth_img'].copy() >0.1,1,0)
        #_difference = np.logical_xor(grd.astype(bool), fake.astype(bool))    #np.where(grd - fake > 0, 1, 0)
        _output = np.zeros(grd.shape)

        _output[fake < grd] = 2
        _output[fake > grd] = 3
        _output[fake * grd > 0] = 1

        orig_imgs = np.copy(orig_copy)
        axs[0, i + (1 * query_ground_truth)].imshow(_output, cmap=custom_cmap, interpolation='nearest')
        axs[1, i + (1 * query_ground_truth)].imshow(
            remain_in_mask(orig_imgs[0], multi_dilation(segmentations[methods[i]], cross, 4)),
            cmap=_cmap)

        if show_titles:
            axs[0, i + (1 * query_ground_truth)].set_title(f"ground truth VS {titles[i]}")

    plt.tight_layout()
    plt.show()
    if input("Exit? [Y]/N ") != "N":
        break  # exit(0)
