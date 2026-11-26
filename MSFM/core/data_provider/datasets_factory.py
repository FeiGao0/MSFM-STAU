# core/data_provider/datasets_factory.py
from torchvision import transforms
from torch.utils.data import DataLoader


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True, is_shuffle=True, mode='train'):
    if dataset == 'mnist':
        from core.data_provider.mm import MovingMNIST
    elif dataset == 'tec':
        from core.data_provider.tec import TECDataset

    if dataset == 'mnist':
        if is_training:
            num_workers = configs.num_workers
            root = 'datasets/mnist/'
        else:
            num_workers = 0
            root = 'datasets/mnist/'
        dataset_obj = MovingMNIST(is_train=is_training,
                                  root=root,
                                  n_frames=20,
                                  num_objects=[2])

    elif dataset == 'tec':
        num_workers = configs.num_workers
        if mode == 'train':
            dataset_obj = TECDataset(path=data_train_path,
                                     input_length=configs.input_length,
                                     total_length=configs.total_length,
                                     train=True, validation=False, test=False)
        elif mode == 'val':
            dataset_obj = TECDataset(path=data_train_path,
                                     input_length=configs.input_length,
                                     total_length=configs.total_length,
                                     train=False, validation=True, test=False)
        else:
            dataset_obj = TECDataset(path=data_test_path,
                                     input_length=configs.input_length,
                                     total_length=configs.total_length,
                                     train=False, validation=False, test=True)

    return DataLoader(dataset_obj,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)