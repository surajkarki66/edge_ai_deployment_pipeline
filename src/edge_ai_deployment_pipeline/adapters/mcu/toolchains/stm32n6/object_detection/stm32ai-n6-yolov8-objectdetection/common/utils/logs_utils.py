# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import csv
import mlflow
from typing import Dict, List
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import tensorflow as tf
import logging

logging.getLogger('tensorflow').setLevel(logging.WARNING)


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    Custom TensorBoard callback that logs the learning rate during training.
    """

    def __init__(self, log_dir: str, **kwargs) -> None:
        '''
        `log_dir` is the directory where the log files will be written.
        '''
        super().__init__(log_dir, **kwargs)
        self.lr_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'metrics'))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        '''
        Write the learning rate to the TensorBoard log file
        '''
        lr = getattr(self.model.optimizer, 'lr', None)
        if lr is not None:
            with self.lr_writer.as_default():
                tf.summary.scalar('learning_rate', lr, step=epoch)
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None) -> None:
        '''
        on_train_end function
        '''
        super().on_train_end(logs)
        self.lr_writer.close()


def mlflow_ini(cfg: DictConfig = None) -> None:
    """
    Initializes MLflow tracking with the given configuration.

    Args:
        cfg (dict): A dictionary containing the configuration parameters for MLflow tracking.

    Returns:
        None
    """
    uri = str(cfg["mlflow"]["uri"])
    if uri.startswith("sqlite://"):
        mlflow.set_tracking_uri(uri)
    elif uri.startswith("file://"):
        mlflow.set_tracking_uri(uri)
    else:
        mlflow.set_tracking_uri("file://" + os.path.abspath(uri))

    general = getattr(cfg, "general", None)
    experiment_name = (
        getattr(general, "project_name", None) if general is not None else None
    ) or "edge_ai_deploy"
    mlflow.set_experiment(experiment_name)

    run_name = HydraConfig.get().runtime.output_dir.split(os.sep)[-1]
    # Explicit start_run avoids Hydra/ClearML autostart races (Run not found).
    mlflow.start_run(run_name=run_name)
    mlflow.set_tag("mlflow.runName", run_name)
    params = {"operation_mode": cfg.operation_mode}
    mlflow.log_params(params)
    mlflow.keras.autolog(log_models=False)


def log_to_file(dir: str, log: str) -> None:
    """
    Appends the given log message to the end of the 'stm32ai_main.log' file in the specified directory.

    Args:
        dir (str): The directory where the log file should be saved.
        log (str): The log message to be written to the file.

    Returns:
        None
    """
    with open(os.path.join(dir, "stm32ai_main.log"), "a") as log_file:
        log_file.write(log + "\n")
        

def log_last_epoch_history(cfg: DictConfig ,output_dir: str) -> None:
    """
    Logs the last epoch history to a file.

    Args:
        output_dir (str): The path to the output directory.

    Returns:
        None
    """
    csv_path = os.path.join(output_dir, cfg.general.logs_dir, "metrics", "train_metrics.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader if row]  # skip empty rows
            metrics = rows[0]
            values = rows[-1]
        log_to_file(output_dir, f'The last epoch history :\n{metrics}\n{values}')
        return values[0]
