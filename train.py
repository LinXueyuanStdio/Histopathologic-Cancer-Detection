import click

from model.MyModel import MyModel
from model.utils.lr_schedule import LRSchedule
from model.utils.Config import Config
from model.pre.data import DataFrameDataset
from model.pre.transforms import trans_train, trans_valid
from model.pre.split_data import generate_split
from model.pre.data_loder import getDataLoder
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


def getTorchDataLoaderByRandomPatch(config):
    labels = pd.read_csv(config.path_label_train)
    train, val = train_test_split(labels, stratify=labels.label, test_size=0.2)
    print("- Train: {}, Val: {}".format(len(train), len(val)))

    dataset_train = DataFrameDataset(df_data=train, data_dir=config.dir_images_train, transform=trans_train)
    dataset_valid = DataFrameDataset(df_data=val, data_dir=config.dir_images_train, transform=trans_valid)

    loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=3)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=config.batch_size//2, shuffle=False, num_workers=3)

    return loader_train, loader_valid, val


def getTorchDataLoaderByWSI(config):
    wsi_path = "data/patch_id_wsi.csv"

    train_ids, cv_ids, train_labels, cv_labels = generate_split(config.path_label_train, wsi_path)
    val = pd.DataFrame()
    val["id"] = cv_ids
    val["label"] = cv_labels

    print(val.shape)

    train_dataset, valid_dataset = getDataLoder(config.dir_images_train, train_ids, cv_ids, train_labels, cv_labels)

    print("- Train: {}, Val: {}".format(len(train_dataset), len(valid_dataset)))

    # Load datasets
    loader_train = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3)
    loader_valid = DataLoader(dataset=valid_dataset, batch_size=config.batch_size//2, shuffle=False, num_workers=3)

    return loader_train, loader_valid, val


@click.command()
@click.option('--data', default="config/data.json",
              help='Path to data json config')
@click.option('--training', default="config/training.json",
              help='Path to training json config')
@click.option('--model', default="config/model.json",
              help='Path to model json config')
@click.option('--output', default="results/local/",
              help='Dir for results and model weights')
@click.option('--gpu', default="1",
              help='cuda:1')
def main(data, training, model, output, gpu):
    # Load configs
    dir_output = output
    config = Config([data, training, model])
    config.device = "cuda:"+gpu
    config.save(dir_output)

    if config.wsi:
        loader_train, loader_valid, val = getTorchDataLoaderByWSI(config)
    else:
        loader_train, loader_valid, val = getTorchDataLoaderByRandomPatch(config)

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
    model.auto_restore()
    model.train(config, loader_train, loader_valid, lr_schedule, val)


if __name__ == "__main__":
    main()
