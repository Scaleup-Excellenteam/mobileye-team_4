import contextlib
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset

from consts import SEQ_IMAG, NAME, IMAG_PATH, GTIM_PATH, JSON_PATH, TRAIN_TEST_VAL, TRAIN, TEST, VALIDATION
from mpl_goodies import plot_rects
import consts


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# TODO: work on this
pd.set_option('display.width', 200, 'display.max_rows', 200,
              'display.max_columns', 200, 'max_colwidth', 40)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class TrafficLightDataSet(Dataset):
    SEQ = 'seq'
    COLOR = 'color'
    LABEL = 'label'
    IMAGE_PATH = 'image_path'
    IMAGE = 'image'
    PREDS = 'preds'
    SCORE = 'score'  # Not really used by this class

    # noinspection PyTypeChecker
    def __init__(self, base_dir, full_image_dir=None, is_train=True, **kwargs):
        """
        This is a data loader. It is used to feed the train and the test
        :param base_dir: Where everything starts...  You should have the following structure underneath:
        base_dir\attention_results\crop_results.h5 - h5 with metadata about the crops
        base_dir\attention_results\attention_results.h5 - h5 with metadata about the full images and attention results
        base_dir\crops - Small png files with the names like in crop_results.h5
        :param full_image_dir: A folder with all full-size images
        :param is_train: True if you want to train, False for test
        :param kwargs: Hmmm... See inside the code
        """
        self.is_train = is_train
        self.full_image_base = full_image_dir  # C.default_train_images
        self.base_dir = base_dir
        self.crop_dir = os.path.join(base_dir, consts.CROP_DIR)
        crops_csv_path = os.path.join(base_dir, consts.ATTENTION_PATH, consts.CROP_CSV_NAME)
        attention_csv_path = os.path.join(base_dir, consts.ATTENTION_PATH, consts.ATTENTION_CSV_NAME)
        crop_data = pd.read_csv(crops_csv_path)  # type: pd.DataFrame
        self.attn_data = pd.read_csv(attention_csv_path)  # type: pd.DataFrame  # To trace back using original picture
        self.attn_data[consts.SEQ] = np.arange(len(self.attn_data))
        train_ratio = kwargs.get('train_ratio', 0.8)  # Amount of train data in the whole data
        with temp_seed(0):
            is_train = np.random.random(len(crop_data)) <= train_ratio

        ignore_ignore = (~crop_data[consts.IS_IGNORE]) & kwargs.get('ignore_ignore', True)

        self.crop_data = crop_data.loc[(is_train == self.is_train) & ignore_ignore]

        limit = kwargs.get('limit', 0)
        if limit > 0:
            # Make things faster if we don't care about the results...
            self.crop_data = self.crop_data.iloc[:limit]

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            assert False, "What to do??"

        row = self.crop_data.iloc[idx]
        image_path = os.path.join(self.crop_dir, row[consts.PATH])
        image = np.array(Image.open(image_path))  # Torch.read_image ?
        return {self.IMAGE: image, self.LABEL: row[consts.IS_TRUE], self.SEQ: row[self.SEQ],
                self.IMAGE_PATH: image_path}

    def get_num_tif(self):
        """
        Get True, Ignore, False counts.
        """
        is_t = self.crop_data[consts.IS_TRUE] == True
        is_f = ~is_t
        is_i = self.crop_data[consts.IS_IGNORE] == True
        # noinspection PyUnresolvedReferences
        return (is_t & ~is_i).sum(), is_i.sum(), (is_f & ~is_i).sum()


class MyNeuralNetworkBase(nn.Module):
    def __init__(self, **kwargs):
        self.w = kwargs.get('w', consts.DEFAULT_CROPS_W)
        self.h = kwargs.get('h', consts.DEFAULT_CROPS_H)
        self.num_in_channels = kwargs.get('num_in_channels', 3)  # RGB
        super(MyNeuralNetworkBase, self).__init__()
        self.name = kwargs.get('name', 'my_first_net')
        self.layers = None
        self.loss_func = None
        self.net = None
        self.set_net_and_loss()

    def set_net_and_loss(self):
        # Feel free to inherit this class and override this function.
        # Here are some totally useless layers. See what YOU need!
        self.layers = (
            # First Convolutional Layer
            nn.Conv2d(self.num_in_channels, 32, kernel_size=3, padding=1),  # 32 filters
            nn.ReLU(),

            # Max pooling
            nn.MaxPool2d(2),

            # Second Convolutional Layer
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Flattening the tensor
            nn.Flatten(),

            # Fully Connected Layer
            nn.Linear(10304, 128),  # Depending on the image size, adjust the linear layer size
            nn.ReLU(),

            # Output Layer
            nn.Linear(128, 1)
        )

        # This is the recommended loss:
        self.loss_func = nn.BCEWithLogitsLoss

        if "The troubles I had getting to a working net" == "many":
            # Here is a sample code to help you debug your layers setup.
            # You already know... Set a breakpoint above, and evaluate this:
            x1 = torch.zeros((1, 3, self.h, self.w))
            all_shapes = [f"{x1.detach().numpy().shape} -->"]
            for ctr, l in enumerate(self.layers):
                print(F"At layer {ctr}: {l}")
                x1 = l(x1)
                all_shapes.append(f"{l._get_name()} --> {x1.detach().numpy().shape}")

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        # This is called both during train, and in evaluation
        logits = self.net(x.transpose(1, 3))
        return logits

    def pred_to_score(self, logits):
        # This is called to see the "real" score. Remember the nn.BCEWithLogitsLoss has a sigmoid at the entrance, so
        # the output needs to go through such too
        if self.loss_func is nn.BCEWithLogitsLoss:
            with torch.no_grad():
                return nn.Sigmoid()(logits)
        return logits


class ModelManager:
    """
    This class helps to create, save, load models. That's all.
    """

    MODEL_STATE = 'model_state'
    METADATA = 'metadata'
    CREATION_KWARGS = 'creation_kwargs'

    @classmethod
    def make_empty_model(cls, factory=None, **kwargs) -> MyNeuralNetworkBase:
        """
        Create an uninitialized model. This is also required when loading a model
        :param factory: Something like MyNeuralNetworkBase
        :param kwargs: Parameters you pass to the factory's __init__ function
        :return: The model itself.
        """
        # Define a model. Also need to do it when loading an existing model.
        if factory is None:
            factory = MyNeuralNetworkBase
        model = factory(**kwargs).to(device)
        model.creation_kwargs = kwargs
        print(f"Model is:\n{model}\n")
        print(f"Loss is:\n{model.loss_func}")
        return model

    @classmethod
    def save_model(cls, model, log_dir, metadata=None, suffix='') -> str:
        """
        Save the model. Return what you need to send to load_model.

        :param model: What you trained
        :param log_dir: Output folder of everything of this model
        :param metadata: Any additional info you want to attach to the model
        :param suffix: Like iteration number or so
        :return: What you need to send to load_model
        """
        to_save = {cls.MODEL_STATE: model.to(torch.device('cpu')).state_dict(),
                   cls.METADATA: metadata,
                   cls.CREATION_KWARGS: model.creation_kwargs,
                   }
        model_path = os.path.abspath(cls.get_model_filename(log_dir, suffix))
        with open(model_path, 'wb') as fh:
            pickle.dump(to_save, fh)
        print(F"Model saved to {model_path}")
        return os.path.abspath(model_path)

    @classmethod
    def load_model(cls, model_path_or_dir, factory=None) -> MyNeuralNetworkBase:
        """
        Load the trained model.

        :param model_path_or_dir: If a directory, will look for model.pkl in there, otherwise full path to model
        :param factory: The same factory as you passed in make_empty_model
        :return: The trained model
        """
        if os.path.isdir(model_path_or_dir):
            model_path = cls.get_model_filename(model_path_or_dir)  # Load final model from directory
        else:
            model_path = model_path_or_dir  # Probably some mid-train model
        to_load = pickle.load(open(model_path.replace('\\', '/'), 'rb'))
        metadata = to_load[cls.METADATA]
        creation_kwargs = to_load[cls.CREATION_KWARGS]
        model = cls.make_empty_model(factory, **creation_kwargs)
        model.load_state_dict(to_load[cls.MODEL_STATE])
        model.metadata = metadata
        return model

    @classmethod
    def get_model_filename(cls, log_dir, suffix=''):
        """
        Generate a model filename.

        :param log_dir: Output folder of everything related to this model.
        :param suffix: Optional suffix for the filename, like iteration number.
        :return: The filename as a string.
        """
        return str(Path(log_dir) / f'model{suffix}.pkl')
