import argparse
import os
import json
import random

import numpy as np
import torch

from brain_trainer import BrainHemorrhageDetection
from settings import Settings


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--seed", default=42)
    cli.add_argument("--device", default='cuda:7')
    cli.add_argument("--data_dir", default='/mnt/dsi_vol1/shaya/rsna-intracranial-hemorrhage-detection/stage_2_train')
    cli.add_argument("--results_dir", default='/home/dsi/shaya/results')
    cli.add_argument("--resume", action='store_true')
    opts = cli.parse_args()


    csv_path = '/home/dsi/shaya/good_slices.csv'
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    with open(os.path.join(opts.results_dir, f'{opts.exp_name}.json')) as json_file:
        setting_dict = json.load(json_file)
    settings = Settings(setting_dict, write_logger=True, exp=opts.exp_name, results_dir=opts.results_dir,resume=opts.resume)
    detector = BrainHemorrhageDetection(
        settings=settings,
        images_dir=opts.data_dir,
        logger=settings.logger,
        csv_path=csv_path, device=opts.device, seed=opts.seed
    )

    detector.train_brain_hemorrhage_detection()


if __name__ == '__main__':
    main()
