import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import consts
from data_utils import TrafficLightDataSet, ModelManager, MyNeuralNetworkBase
from mpl_goodies import nn_examiner_example

from data_utils import device


def run_one_train_epoch(model: MyNeuralNetworkBase, dataset: TrafficLightDataSet, balance_samples: bool = True):
    """
    Go over one batch, and either train the model, of just get the scores
    :param model: The model you train
    :param dataset: Data to work on
    :param balance_samples: As we have much more False than True, we balance them
    :return:
    """
    train_dataset = dataset
    num_tif = train_dataset.get_num_tif()
    t_weight = 1. / num_tif[0]
    f_weight = 1. / num_tif[2]
    weights = torch.tensor(np.where(train_dataset.crop_data[consts.IS_TRUE], t_weight, f_weight))
    sampler = WeightedRandomSampler(weights, len(weights)) if balance_samples else None
    data_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    loss_func = model.loss_func
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    acc_loss = 0
    tot_samples = 0
    for i_batch, batch in enumerate(data_loader):

        imgs = (batch['image'] / 255).to(device)  # Note: imgs is float32
        labs = (batch['label']).to(device).float()

        bs = len(imgs)

        # zero the gradient buffers
        optimizer.zero_grad()

        # predict:
        preds = model(imgs)

        loss = loss_func()(preds.reshape(bs), labs.reshape(bs))

        acc_loss += float(loss.detach()) * bs
        tot_samples += bs

        # (1) propagate the loss, and (2) do the update
        loss.backward()
        optimizer.step()
    return model, -1 if tot_samples == 0 else acc_loss / tot_samples


def run_one_test_epoch(model: MyNeuralNetworkBase, dataset: TrafficLightDataSet) -> (float, dict):
    """
    Go over the batch and calculate the scores and total loss
    :param model: The model you test
    :param dataset: Data to work on
    :return: loss, scores
    """
    test_loader = DataLoader(dataset, batch_size=16)
    loss_func = model.loss_func

    acc_loss = 0
    all_scores_acc = {TrafficLightDataSet.SEQ: [],
                      TrafficLightDataSet.SCORE: [],
                      TrafficLightDataSet.PREDS: [],
                      }
    # no_grad is VERY IMPORTANT when you don't want to train... Much faster, and less memory
    with torch.no_grad():
        for i_batch, batch in enumerate(test_loader):
            imgs = (batch[TrafficLightDataSet.IMAGE] / 255).to(device)  # Note: imgs is float32
            labs = (batch[TrafficLightDataSet.LABEL]).to(device).float()
            bs = len(imgs)
            # predict:
            preds = model(imgs)
            score = model.pred_to_score(preds)
            loss = loss_func()(preds.reshape(bs), labs.reshape(bs))
            acc_loss += float(loss.detach()) * bs
            all_scores_acc[TrafficLightDataSet.SEQ].extend(batch[TrafficLightDataSet.SEQ].tolist())
            all_scores_acc[TrafficLightDataSet.PREDS].extend(preds.numpy().ravel().tolist())
            all_scores_acc[TrafficLightDataSet.SCORE].extend(score.numpy().ravel().tolist())
    tot_samples = len(all_scores_acc[TrafficLightDataSet.SCORE])
    loss = -1 if tot_samples == 0 else acc_loss / tot_samples
    return loss, all_scores_acc


def train_a_model(model: MyNeuralNetworkBase,
                  train_dataset: TrafficLightDataSet,
                  test_dataset: TrafficLightDataSet,
                  log_dir: str,
                  num_epochs: int = 20,
                  ) -> str:
    """
    Do the train loop. Write intermediate results to TB and trained files.
    :param model: The model to train (or continue training)
    :param train_dataset: Dataset of images to train on
    :param test_dataset: Dataset of images to test on
    :param log_dir: Where to store log and mid-train models
    :param num_epochs: How many rounds.. You will eventually need to raise to hundreds
    :return: Filename of last the saved model
    """
    writer = SummaryWriter(log_dir)
    metadata = None
    for ep in range(num_epochs):
        ep_time_start = datetime.datetime.now()
        model, train_loss = run_one_train_epoch(model=model, dataset=train_dataset, balance_samples=True)
        test_loss, test_scores = run_one_test_epoch(model=model, dataset=test_dataset)
        ep_time_end = datetime.datetime.now()

        print(f'Epoch {ep}: train/test: {train_loss}, {test_loss}, '
              f'took {str(ep_time_end - ep_time_start)[2:-3]}')
        writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, ep)
        metadata = {
            'model_name': model.name,
            'num_epochs': ep,
            'last_train_error': train_loss,
            'last_test_error': test_loss,
        }
        if ((ep + 1) % 10) == 0 or ep + 1 == num_epochs:
            # It's a good idea to save the model while training... So you can test it before the loop ends
            ModelManager.save_model(model, log_dir, metadata, suffix=f'_{ep:04}')

    model_path = ModelManager.save_model(model, log_dir, metadata)

    return model_path


def go_train(base_dir, model_name, train_dataset, test_dataset, num_epochs=20):
    """
    Starts the model training process.

    :param base_dir: Base directory containing datasets and other resources.
    :param model_name: Name assigned to the model.
    :param train_dataset: Dataset used for training.
    :param test_dataset: Dataset used for testing/evaluation.
    :param num_epochs: Number of training epochs.
    :return: Path to the trained model.
    """

    model = ModelManager.make_empty_model(name=model_name)
    tb_exe = os.path.join(os.path.split(sys.executable)[0], 'tensorboard')
    log_dir = os.path.join(base_dir, consts.LOGS_DIR, model.name)
    print(f"Run: {tb_exe} --logdir={os.path.abspath(os.path.split(log_dir)[0])}")
    trained_model_path = train_a_model(model, train_dataset, test_dataset, log_dir, num_epochs=num_epochs)
    return trained_model_path


def go_classify(base_dir: str, trained_model_path: str, dataset: TrafficLightDataSet) -> dict:
    """
    Get the results on the dataset using the trained model.

    param base_dir: Base directory containing datasets and other resources.
    param trained_model_path: Path to the trained model file.
    param dataset: Dataset on which the classification is to be performed.
    return: Dictionary with keys as defined in run_one_test_epoch.
             The keys include TrafficLightDataSet.SEQ, TrafficLightDataSet.SCORE, and TrafficLightDataSet.PREDS.
    """

    print(f"Trying to load from {os.path.abspath(trained_model_path)}")
    model = ModelManager.load_model(trained_model_path)
    test_loss, test_scores = run_one_test_epoch(model, dataset)
    print(f"Total loss is {test_loss}")
    return test_scores


def examine_my_results(base_dir,
                       full_images_dir,
                       trained_model_path,
                       dataset,
                       scores_h5_filename=None,
                       ):
    """
    Visualizes the results of the trained model.

    :param base_dir: Base directory containing datasets and other resources.
    :param full_images_dir: Directory containing full-sized images.
    :param trained_model_path: Path to the trained model.
    :param dataset: Dataset to evaluate and visualize results.
    :param scores_h5_filename: Filename for storing the scores in h5 format. If None, a default name will be generated.
    :return: None
    """

    model = ModelManager.load_model(trained_model_path)
    scores = go_classify(base_dir, trained_model_path, dataset)

    # Set up things for the viewer:
    def update_path(df, prev_col, new_col, prefix):
        """
        Updates paths in a DataFrame column with a specified prefix and saves them in a new column.

        :param df: Input DataFrame.
        :param prev_col: Column name in df that contains the paths to be updated.
        :param new_col: Column name where the updated paths will be stored.
        :param prefix: Prefix (directory path) to be added to each path in prev_col.
        :return: DataFrame with updated paths in new_col.
        """
        df2 = df.copy()
        series = df2.pop(prev_col)
        values = [os.path.abspath(os.path.join(prefix, f)) for f in series]
        df2[new_col] = values
        return df2
    crop_data = update_path(dataset.crop_data, 'path', 'crop_path', dataset.crop_dir)
    full_data = update_path(dataset.attn_data, 'path', 'full_path', full_images_dir)

    # Ok, from here on, some pandas tricks...
    results = pd.DataFrame(scores) \
        .merge(crop_data, on=TrafficLightDataSet.SEQ) \
        .merge(full_data.drop(consts.COL, axis=1), on=TrafficLightDataSet.SEQ)

    if scores_h5_filename is None:
        scores_h5_filename = os.path.join(os.path.split(trained_model_path)[0], 'scores.h5')
    with pd.HDFStore(scores_h5_filename, mode='w') as fh:
        fh['data'] = results
        fh['metadata'] = pd.Series({'model_dir': trained_model_path,
                                    'crop_dir': dataset.crop_dir,
                                    'full_dir': full_images_dir,
                                    'name': model.name,
                                    })

    nn_examiner_example(scores_h5_filename)


def main():
    base_dir = consts.DATA_DIR
    full_images_dir = os.path.join(base_dir, consts.FULL_IMAGES_DIR)
    model_name = 'my_model_final_2'
    train_dataset = TrafficLightDataSet(base_dir, full_images_dir, is_train=True)
    test_dataset = TrafficLightDataSet(base_dir, full_images_dir, is_train=False)
    trained_model_path = go_train(base_dir, model_name, train_dataset, test_dataset, num_epochs=10)
    examine_my_results(base_dir, full_images_dir, trained_model_path, test_dataset)


if __name__ == '__main__':
    main()
