from trainer import train
import os


def main():
    runs = []
    # only try this when on a powerful machine
    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'original', 'n_rounds': 10})
    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'original', 'n_rounds': 10})
    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'original', 'n_rounds': 10})

    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'growing', 'n_rounds': [40,20,15,10]})
    runs.append({'dataset': 'DRIVE', 'data_rotation': 3, 'model': 'growing', 'n_rounds': [40,9,8,7]})# make sudden steep

    # for weaker machines
    # runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'original', 'n_rounds': 10})

    # runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'growing', 'n_rounds': 10})
    # runs.append({'dataset': 'DRIVE', 'data_rotation': 180, 'model': 'growing', 'n_rounds': [40, 20, 15, 10]})

    for run in runs:
        print(f"*** starting run: {run} ***")
        dataset = run['dataset']
        img_out_dir = os.path.join("data", "{}".format(dataset))
        train(run, img_out_dir)



if __name__ == '__main__':
    main()
