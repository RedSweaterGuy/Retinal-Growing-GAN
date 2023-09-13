import numpy as np
import os
import json
import glob
from json import JSONEncoder
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


methods = ['origGANnew', 'GrowConst10', 'growGANvarying', 'growGANconstant']
img_size = (640, 640)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def create_img_fundus_segmentation_overlay(orig_img, cmap, ground_truth_img, cross):
    fig_poster, axes_poster = plt.subplots(1, 3, figsize=(15, 6))
    ax_poster = axes_poster[0]
    ax_poster.imshow(orig_img)
    ax_poster.text(0.5, -0.1, "Fundus", transform=ax_poster.transAxes, ha='center', fontsize=24)
    ax_poster.axis('off')

    ax_poster = axes_poster[1]
    ax_poster.imshow(ground_truth_img, cmap=cmap)
    ax_poster.text(0.5, -0.1, "Segmentation", transform=ax_poster.transAxes, ha='center', fontsize=24)
    ax_poster.axis('off')

    ax_poster = axes_poster[2]
    ax_poster.imshow(remain_in_mask(orig_img, multi_dilation(ground_truth_img, cross, 4)), cmap=cmap)
    ax_poster.text(0.5, -0.1, "Overlay", transform=ax_poster.transAxes, ha='center', fontsize=24)
    ax_poster.axis('off')

    plt.tight_layout()
    plt.savefig("for_poster.png")
    plt.close()


def apply_colors(color1, color2, values):
    return (color1[0] * values + color2[0] * (1 - values)), (color1[1] * values + color2[1] * (1 - values)), (
            color1[0] * values + color2[0] * (1 - values))


def remain_in_mask(img, masks):
    img[masks < 0.2] = 0
    return img


def multi_dilation(image, kernel, iterations):
    for i in range(iterations):
        image = dilation(image, kernel)
    return image


def load_models():
    f_model = "pretrained/{}/{}_best.json"
    f_weights = "pretrained/{}/{}_best.h5"
    dataset = "DRIVE"
    models = []
    for i in range(len(methods)):
        # load the model and weights
        with open(f_model.format(dataset, methods[i]), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(f_weights.format(dataset, methods[i]))
        generated = model.predict(np.zeros((1, img_size[0], img_size[1], 3)), batch_size=1)
        models.append(model)
    return models

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    show_titles = True
    rotate_by = 0
    query_ground_truth = True
    _cmap = "gray"

    custom_color_false_positive = (197, 27, 125)
    custom_color_false_negative = (77, 146, 33)
    custom_color_true_positive = (0, 0, 0)#(255, 255, 255)
    background_color = (255, 255, 255)#(0, 0, 0)

    titles = ['Original GAN','GrowingGAN (10R)', 'GrowingGAN (10,4,4,4R)', 'GrowingGAN (5R)']

    batch_size = 1

    cross = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

    # Flow:
    # ask for image
    # make segmentation with the three models
    # output results with matplotlib
    generate_with_models = False
    if generate_with_models:
        from keras.models import model_from_json
        from skimage.transform import resize
        from skimage.morphology import dilation
    generate_for_all = generate_with_models and False
    models = load_models() if generate_with_models else None

    def convert_to_np(data):
        out = dict()
        for key in data:
            out[key] = dict()
            for arr_key in data[key]:
                out[key][arr_key] = np.asarray(data[key][arr_key])
        return out


    def load_images(image_path):
        fparts = os.path.split(image_path)
        with open(os.path.join(fparts[0], os.pardir, "json", f"{fparts[1].split('.')[0]}.json"), "r") as f:
            data = convert_to_np(json.load(f))
            return data['overlays'], data['segmentations'], data['diff_imgs']


    def get_images(image_path):
        overlays = dict()
        segmentations = dict()  # dict where all segmentations for one run will be stored
        diff_imgs = dict()

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
        ground_truth_img = np.asarray(ground_truth_img).astype(np.float32) / 255
        ground_truth_img = resize(ground_truth_img, img_size)
        segmentations['ground_truth_img'] = ground_truth_img

        overlays['ground_truth'] = remain_in_mask(orig_imgs[0], multi_dilation(ground_truth_img, cross, 4))
        for i in range(len(methods)):
            generated = models[i].predict(test_imgs, batch_size=batch_size)
            generated = np.squeeze(generated, axis=3)[0]
            segmentations[methods[i]] = generated
            orig_imgs = np.copy(orig_copy)
            overlays[methods[i]] = remain_in_mask(orig_imgs[0], multi_dilation(generated, cross, 4))

            fake = segmentations[methods[i]].copy()
            grd = segmentations['ground_truth_img'].copy()
            _output = grd - fake

            diff_img = np.zeros((_output.shape[0], _output.shape[1], 3), dtype=int)
            _output[np.logical_and(_output > 0, _output < 0.1)] = 0
            _output[np.logical_and(_output < 0, _output > -0.1)] = 0

            diff_img[_output < 0, :] = np.transpose(
                apply_colors(custom_color_false_positive, background_color, abs(_output[_output < 0])))

            diff_img[_output > 0, :] = np.transpose(
                apply_colors(custom_color_false_negative, background_color, abs(_output[_output > 0])))

            diff_img[np.logical_and(_output == 0, grd > 0.1), :] = custom_color_true_positive
            diff_img[np.logical_and(_output == 0, grd <= 0.1), :] = background_color

            diff_imgs[methods[i]] = diff_img

        fparts = os.path.split(image_path)
        data = dict()
        data["overlays"] = overlays
        data["segmentations"] = segmentations
        data["diff_imgs"] = diff_imgs
        json_folder = os.path.join(fparts[0], os.pardir, "json")
        if not os.path.exists(json_folder):
            os.makedirs(json_folder, exist_ok=True)
        with open(os.path.join(json_folder, f"{fparts[1].split('.')[0]}.json"), "w") as f:
            json.dump(data, f, cls=NumpyArrayEncoder)

        return overlays, segmentations, diff_imgs


    def open_file_dialog():
        image_path = filedialog.askopenfilename(title="Choose a Fundus Image")
        if not image_path:
            return

        # first plot
        figs, axs = plt.subplots(3, len(methods) + (1 * query_ground_truth), sharex=True, sharey=True)
        for ax in axs.ravel():
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])


        if generate_for_all:
            fparts = os.path.split(image_path)
            for f in glob.glob(os.path.join(fparts[0], "*.tif")):
                get_images(f)
        overlays, segmentations, diff_imgs = load_images(image_path) if models is None else get_images(image_path)

        axs[0, 0].imshow(segmentations['ground_truth_img'], cmap=_cmap)
        axs[2, 0].imshow(segmentations['ground_truth_img'], cmap='gray_r')
        if show_titles:
            axs[2,0].set_title("Ground truth")

        # Uncomment to create the image of the poster, with fundus image, segmentation and overlay
        #create_img_fundus_segmentation_overlay(orig_imgs[0], _cmap, ground_truth_img, cross)

        axs[1, 0].imshow(overlays['ground_truth'], cmap=_cmap)
        if show_titles:
            axs[0, 0].set_title("Ground truth")

        for i in range(len(methods)):
            axs[0, i + (1 * query_ground_truth)].imshow(segmentations[methods[i]], cmap=_cmap)
            axs[1, i + (1 * query_ground_truth)].imshow(overlays[methods[i]], cmap=_cmap)

            if show_titles:
                axs[0, i + (1 * query_ground_truth)].set_title(titles[i])

            axs[2, i + (1 * query_ground_truth)].set_facecolor('white')

            axs[2, i + (1 * query_ground_truth)].imshow(diff_imgs[methods[i]])

            if show_titles:
                axs[2, i + (1 * query_ground_truth)].set_title(f"GT VS {titles[i]}")
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
        return

    app = tk.Tk()
    app.title("Vizualization Demo for Growing GAN of Retinal Vessel Segmentation")
    app.geometry("600x100")

    open_button = tk.Button(app, text="Choose Fundus Image", command=open_file_dialog)
    open_button.pack()

    app.mainloop()
    return


if __name__ == "__main__":
    main()
