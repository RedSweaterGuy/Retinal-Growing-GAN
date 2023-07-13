from datagen import gen_data
from trainer import train


def main():
    runs = []
    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'original'})
    #runs.append({'dataset': 'DRIVE', 'data_rotation': 36, 'model': 'original'})
    #runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'original'})
    for run in runs:
        print(f"starting run: {run}")
        img_out_dir = gen_data(run)
        train(run, img_out_dir)



if __name__ == '__main__':
    main()
