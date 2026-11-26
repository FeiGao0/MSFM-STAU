import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TECDataset(Dataset):
    def __init__(self, path, input_length, total_length, train=True, validation=False, test=False, year=None):
        super(TECDataset, self).__init__()
        print("*****Start collecting data*****")
        self.path = path
        self.train = train
        self.validation = validation
        self.test = test
        self.input_length = input_length
        self.predict_step = total_length - input_length
        self.year = year

        self.TEC_2009_path = os.path.join(self.path, "TEC_2009.npy")
        self.TEC_2010_path = os.path.join(self.path, "TEC_2010.npy")
        self.TEC_2011_path = os.path.join(self.path, "TEC_2011.npy")
        self.TEC_2012_path = os.path.join(self.path, "TEC_2012.npy")
        self.TEC_2013_path = os.path.join(self.path, "TEC_2013.npy")
        self.TEC_2014_path = os.path.join(self.path, "TEC_2014.npy")
        self.TEC_2015_path = os.path.join(self.path, "TEC_2015.npy")
        self.TEC_2018_path = os.path.join(self.path, "TEC_2018.npy")
        self.TEC_2019_path = os.path.join(self.path, "TEC_2019.npy")
        self.TEC_2020_path = os.path.join(self.path, "TEC_2020.npy")
        self.TEC_2023_path = os.path.join(self.path, "TEC_2023.npy")
        self.TEC_2024_path = os.path.join(self.path, "TEC_2024.npy")

        def load_and_norm(fpath):
            data = np.load(fpath) / 10.0
            return (data - 0) / (200 - 0)

        self.TEC_2009 = load_and_norm(self.TEC_2009_path)
        self.TEC_2010 = load_and_norm(self.TEC_2010_path)
        self.TEC_2011 = load_and_norm(self.TEC_2011_path)
        self.TEC_2012 = load_and_norm(self.TEC_2012_path)
        self.TEC_2013 = load_and_norm(self.TEC_2013_path)
        self.TEC_2014 = load_and_norm(self.TEC_2014_path)
        self.TEC_2015 = load_and_norm(self.TEC_2015_path)
        self.TEC_2018 = load_and_norm(self.TEC_2018_path)
        self.TEC_2019 = load_and_norm(self.TEC_2019_path)
        self.TEC_2020 = load_and_norm(self.TEC_2020_path)
        self.TEC_2023 = load_and_norm(self.TEC_2023_path)
        self.TEC_2024 = load_and_norm(self.TEC_2024_path)

        train_data = np.concatenate([self.TEC_2009, self.TEC_2010, self.TEC_2011,
                                     self.TEC_2012, self.TEC_2013, self.TEC_2014], axis=0)
        train_data = train_data.reshape(-1, 71, 73)

        val2015 = self.TEC_2015.reshape(-1, 71, 73)
        pre2014 = train_data[-self.input_length:]
        val2015 = np.concatenate([pre2014, val2015], axis=0)

        val2019 = self.TEC_2019.reshape(-1, 71, 73)
        pre2018 = self.TEC_2018.reshape(-1, 71, 73)[-self.input_length:]
        val2019 = np.concatenate([pre2018, val2019], axis=0)

        test2020 = self.TEC_2020.reshape(-1, 71, 73)
        pre2019 = self.TEC_2019.reshape(-1, 71, 73)[-self.input_length:]
        test2020 = np.concatenate([pre2019, test2020], axis=0)

        test2024 = self.TEC_2024.reshape(-1, 71, 73)
        pre2023 = self.TEC_2023.reshape(-1, 71, 73)[-self.input_length:]
        test2024 = np.concatenate([pre2023, test2024], axis=0)

        if self.train:
            x, y = self.split_data(train_data, self.input_length, self.predict_step)
        elif self.validation:
            x1, y1 = self.split_data(val2015, self.input_length, self.predict_step)
            x2, y2 = self.split_data(val2019, self.input_length, self.predict_step)
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
        else:
            if self.year == '2020':
                x, y = self.split_data(test2020, self.input_length, self.predict_step)
            elif self.year == '2024':
                x, y = self.split_data(test2024, self.input_length, self.predict_step)
            else:
                x1, y1 = self.split_data(test2020, self.input_length, self.predict_step)
                x2, y2 = self.split_data(test2024, self.input_length, self.predict_step)
                x = np.concatenate([x1, x2], axis=0)
                y = np.concatenate([y1, y2], axis=0)

        data = np.concatenate([x, y], axis=1)
        data = data[:, :, np.newaxis, :, :]
        self.data = torch.from_numpy(data).float()
        self.len = len(self.data)
        print(f"Data loading completed，with a total of {self.len} samples，each with a shape of {self.data[0].shape}")
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def split_data(self, data, time_step, predict_step):
        x, y = [], []
        for i in range(0, data.shape[0] - time_step - predict_step + 1, predict_step):
            x.append(data[i:i+time_step, :, :])
            y.append(data[i+time_step:i+time_step+predict_step, :, :])
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        return x, y