from datagen import gen_data
from trainer import train
import os


def main():
    runs = []
    #runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'original', 'n_rounds': 10})
    #runs.append({'dataset': 'DRIVE', 'data_rotation': 36, 'model': 'original', 'n_rounds': 10})
    #runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'original', 'n_rounds': 10})
    runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'growing', 'n_rounds': 10})
    for run in runs:
        print(f"starting run: {run}")
        #img_out_dir = gen_data(run)
        dataset = run['dataset']
        #step_size = run['data_rotation']
        img_out_dir = os.path.join("data", "{}".format(dataset))
        train(run, img_out_dir)



if __name__ == '__main__':
    main()
