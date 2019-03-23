import click

from model.MyModel import MyModel
from model.utils.lr_schedule import LRSchedule
from model.utils.Config import Config
from model.pre.data import DataFrameDataset, trans_train, trans_valid
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
    config.dir_answers = dir_output + "formulas_test/"

    # Load datasets
#     train_ids, cv_ids, train_labels, cv_labels = generate_split(train_label_path, wsi_path)
    labels = pd.read_csv(config.path_label_test)
    print("- There are {} items to test.".format(labels.shape[0]))

    dataset_valid = DataFrameDataset(df_data=labels, data_dir=config.dir_images_test, transform=trans_valid)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=config.batch_size//2, shuffle=False, num_workers=3)

    # Build model and train
    model = MyModel(config, dir_output)
    model.build_pred(config)
    model.restore()
    model.evaluate(config, loader_valid, labels)


if __name__ == "__main__":
    main()