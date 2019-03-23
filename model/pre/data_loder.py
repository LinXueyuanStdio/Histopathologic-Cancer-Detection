from model.pre.data import ImageDataset, LabelDataset, MainDataset, trans_test
from model.pre.split_data import generate_split
# from data import ImageDataset, LabelDataset, MainDataset, trans_test
# from split_data import generate_split
import numpy as np


def getDataLoder(image_path, train_images, valid_images, train_labels, valid_labels):
    '''
    Args:
        train_images: ['4faf2b15c22cc437fb16fb11b2f59f2a44afb88c', '67b241cf6dfb3e486bea7ca652500e59783eb124', '0e0a49f00652b915dd6afc661c9bee4eb7f75723', 'a912e404d73397e65f7bfd4630f2a13ad800fe47', 'ba5ba4d9622b91f44a3f1bca709c7a7344b5bf78']
        valid_images: ['a0e35f55df27bfe6a43894780333f202108da95c', '781a505340288d069b646d63eb0897528cecf752', 'efcf1f070efa0100968773c522af474f561cee40', '381e7a49d67485b2cb084b6e5134a71443d44b5d', 'a08962087ec1477dbec7cdf48a2d13c5d0070cbd']
        train_labels: [0, 1, 0, 1, 0]
        valid_labels: [0, 0, 0, 0, 0]
    return:
        train_dataset, valid_dataset
    '''
    train_images = [image_path+"{}.tif".format(i) for i in train_images]
    valid_images = [image_path+"{}.tif".format(i) for i in valid_images]
    train_labels = np.asarray(train_labels).reshape(-1, 1)
    valid_labels = np.asarray(valid_labels).reshape(-1, 1)
    train_images_dataset = ImageDataset(train_images)
    valid_images_dataset = ImageDataset(valid_images)
    train_labels_dataset = LabelDataset(train_labels)
    valid_labels_dataset = LabelDataset(valid_labels)

    train_dataset = MainDataset(train_images_dataset, train_labels_dataset, trans_test)
    valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, trans_test)

    return train_dataset, valid_dataset

if __name__ == "__main__":
    data_dir = '/mnt/d/project/AI.Health.kaggle/'
    label_dir = '../../data/'
    train_path = data_dir+'train/'
    test_path = data_dir+'test/'
    train_label_path = label_dir+'full_train_labels.csv'
    wsi_path = label_dir+'patch_id_wsi.csv'

    train_ids, cv_ids, train_labels, cv_labels = generate_split(train_label_path, wsi_path)

    train_dataset, valid_dataset = getDataLoder(train_path, train_ids, cv_ids, train_labels, cv_labels)

    print(len(train_dataset), len(valid_dataset))

    for i in train_dataset:
        print(i)
        break