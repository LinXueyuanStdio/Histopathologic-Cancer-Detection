import os


def format_labels_for_dataset(labels):
    '''
    pd.DataFrame -> np.array
    '''
    return labels['label'].values.reshape(-1, 1)


def format_path_to_images_for_dataset(labels, path):
    '''
    pd.DataFrame, str -> List
    '''
    return [os.path.join(path, '{}.tif'.format(f)) for f in labels['id'].values]
