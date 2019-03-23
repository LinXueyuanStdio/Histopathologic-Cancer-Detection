import click

from model.MyModel import MyModel
from model.utils.lr_schedule import LRSchedule
from model.utils.Config import Config
from model.pre.data import DataFrameDataset
from model.pre.transforms import trans_train, trans_valid
from model.pre.split_data import generate_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

@click.command()
@click.option('--data', default="config/data.json",
        help='Path to data json config')
@click.option('--training', default="config/training.json",
        help='Path to training json config')
@click.option('--model', default="config/model.json",
        help='Path to model json config')
@click.option('--output', default="results/local/",
        help='Dir for results and model weights')
def main(data, training, model, output):
    # Load configs
    dir_output = output
    config = Config([data, training, model])
    config.save(dir_output)

    # Load datasets
#     train_ids, cv_ids, train_labels, cv_labels = generate_split(train_label_path, wsi_path)
    labels = pd.read_csv(config.path_label_train)
    train, val = train_test_split(labels, stratify=labels.label, test_size=0.2)
    print(len(train), len(val))

    dataset_train = DataFrameDataset(df_data=train, data_dir=config.dir_images_train, transform=trans_train)
    dataset_valid = DataFrameDataset(df_data=val, data_dir=config.dir_images_train, transform=trans_valid)

    loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=3)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=config.batch_size//2, shuffle=False, num_workers=3)

    # Define learning rate schedule
    n_batches_epoch = len(loader_train)
    lr_schedule = LRSchedule(lr_init=config.lr_init,
            start_decay=config.start_decay*n_batches_epoch,
            end_decay=config.end_decay*n_batches_epoch,
            end_warm=config.end_warm*n_batches_epoch,
            lr_warm=config.lr_warm,
            lr_min=config.lr_min)

    # Build model and train
    model = MyModel(config, dir_output)
    model.build_train(config)
    model.restore()
    model.train(config, loader_train, loader_valid, lr_schedule)


if __name__ == "__main__":
    main()