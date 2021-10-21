import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
import wandb as wandb
from torch.utils.data import DataLoader

import time
import json
import pydicom
import pandas as pd

from tqdm import trange, tqdm

from rsna_dataset import RSNADataset
from classification_model import classification_model,CombinedModel

cudnn.benchmark = True


class BrainHemorrhageDetection(object):
    def __init__(self, settings, images_dir, logger, csv_path, device, seed):
        """
        This method initializes settings
        :param settings: application settings
        """
        self.logger = logger
        self.settings = settings
        self.images_dir = images_dir
        self.device = device

        if settings.combined_net:
            self.net = CombinedModel(
                encoder_name=self.settings.encoder_name,
                encoder_depth=self.settings.encoder_depth,
                encoder_weights=self.settings.encoder_weights,
                decoder_channels=self.settings.decoder_channels,
                in_channels=self.settings.n_channels,
                classes=self.settings.classes,
                activation='sigmoid',device=device
            )
        else:
            self.net = classification_model(
                encoder_name=self.settings.encoder_name,
                encoder_depth=self.settings.encoder_depth,
                encoder_weights=self.settings.encoder_weights,
                decoder_channels=self.settings.decoder_channels,
                in_channels=self.settings.n_channels,
                classes=self.settings.classes,
                activation='sigmoid'
            )
        self.net.to(self.device)
        self.df = pd.read_csv(csv_path)
        self.labels_dict = {
            'intraparenchymal': 1,
            'epidural': 2,
            'intraventricular': 3,
            'subarachnoid': 4,
            'subdural': 5,
            'normal': 0
        }

        partition, labels = self.get_partition_labels(
            frac=0.25,
            seed=seed
        )
        self.logger.info('# training samples: {} # validation samples: {} # test samples: {}'.format(
            len(partition['train']), len(partition['validation']), len(partition['test'])
        ))

        train_dl = RSNADataset(
            partition['train'], self.images_dir, self.settings, labels, use_smaller_datasize=False
        )
        val_dl = RSNADataset(
            partition['validation'], self.images_dir, self.settings, labels, use_smaller_datasize=False, train=False
        )

        self.train_loader = DataLoader(train_dl, batch_size=self.settings.batch_size, shuffle=True, drop_last=True)

        self.val_loader = DataLoader(val_dl, batch_size=self.settings.batch_size, shuffle=False)
        weights = self.get_weights(partition, labels)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = torch.optim.Adam(self.net.parameters_to_grad(), lr=self.settings.initial_learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.settings.lr_decay_step_size,
                                                         gamma=self.settings.gamma_decay)
        self.start_epoch = 0
        self.step = 0

        if settings.resume:
            state_dict = torch.load(os.path.exists(os.path.join(settings.checkpoint_dir, 'final_ckpt.pt')))
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.start_epoch = state_dict['epoch']
            self.scheduler.load_state_dict(state_dict['sched_state_dict'])
            self.net.load_state_dict(state_dict['model'])
            self.step = state_dict['step']
            # support resume wandb graph easily
            wandb.init(
                project="transfer_avg",
                id=state_dict['run_id'],
                resume=state_dict['run_id'],
                settings=wandb.Settings(start_method="fork"),
                name=settings.exp,
            )
        else:
            wandb.init(
                project="transfer_avg",
                settings=wandb.Settings(start_method="fork"),
                name=settings.exp,
            )

    def get_metadata_from_dicom(self, img_dicom):
        metadata = {
            "window_center": img_dicom.WindowCenter,
            "window_width": img_dicom.WindowWidth,
            "intercept": img_dicom.RescaleIntercept,
            "slope": img_dicom.RescaleSlope,

        }
        return {k: RSNADataset.get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

    def window_image(self, img, window_center, window_width, intercept, slope):
        img = img * slope + intercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        # img_min = self.window_center - self.window_width // 2
        # img_max = self.window_center + self.window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    def get_partition_labels(self, frac, seed):

        if not os.path.exists(os.path.join(self.settings.simulation_folder, 'partition.json')):

            partition = dict()
            labels = dict()
            df = self.df.sample(frac=frac, random_state=seed)
            df = df.replace(np.nan, 0)

            for key in self.labels_dict.keys():
                df = df.replace(key, self.labels_dict[key])

            df1 = df.applymap(np.isreal)
            df = df[df1.labels == True]
            df = pd.concat([df[df.labels == 0][:4000], df[df.labels != 0]])
            lbls = []
            ids = []
            self.logger.info(len(df))
            for i in trange(len(df)):
                id_ = df.id.iloc[i]
                img_path = os.path.join(self.images_dir, 'ID_{}.dcm'.format(id_))
                img_dicom = pydicom.read_file(img_path)
                metadata = self.get_metadata_from_dicom(img_dicom)
                img = self.window_image(img_dicom.pixel_array, **metadata)

                if img.max() > img.min():
                    # remove blank images
                    label = df.iloc[i, 1]
                    labels[id_] = label
                    lbls.append(label)
                    ids.append(id_)

            new_df = pd.DataFrame(np.array([ids, lbls]).transpose(),
                                  columns=["id", "labels"])

            training = new_df.sample(frac=0.6, random_state=seed)
            validation = new_df.drop(training.index, axis=0)
            test = validation.sample(frac=0.5, random_state=seed)
            validation = validation.drop(test.index, axis=0)
            partition['train'] = list(training.id)
            partition['validation'] = list(validation.id)
            partition['test'] = list(test.id)

            partition_json_path = os.path.join(self.settings.simulation_folder, 'partition.json')
            with open(partition_json_path, 'w') as f:
                json.dump(partition, f, indent=4)

            labels_json_path = os.path.join(self.settings.simulation_folder, 'labels.json')
            with open(labels_json_path, 'w') as f:
                json.dump(labels, f, indent=4)
        else:
            with open(os.path.join(self.settings.simulation_folder, 'partition.json')) as json_file:
                partition = json.load(json_file)

            with open(os.path.join(self.settings.simulation_folder, 'labels.json')) as json_file:
                labels = json.load(json_file)

        return partition, labels

    def get_weights(self, partition, labels):
        training_labels = []
        weights = []
        for id in partition['train']:
            training_labels.append(labels[id])

        num_samples = len(training_labels)
        training_labels = np.array(training_labels)
        for i in range(6):
            cur_class_num_samples = np.sum(training_labels == i)
            weights.append(round(num_samples / cur_class_num_samples, 2))

        return torch.tensor(weights).to(self.device).float()

    def trainval(self, is_train, dl):
        loss_cur = []
        acc_cur = []
        if is_train:
            self.net.train()  # Set model to training mode
            descriptor = 'train'
        else:
            self.net.eval()
            descriptor = 'eval'
        with torch.set_grad_enabled(is_train):
            pbar =tqdm(enumerate(dl), total=len(dl))

            for i, data in pbar:
                running_corrects = 0
                inputs = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                batch_size = inputs.size(0)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    self.step+=1

                loss_cur.append(loss.item())
                # preds = preds.squeeze()
                running_corrects += torch.sum(preds == labels.data).item()
                acc_cur.append(running_corrects / batch_size)

                if i % 100 == 0 or i == len(dl) -1:
                    pbar.set_description(f'{descriptor} loss: {np.mean(loss_cur)} {descriptor} accuracy: {np.mean(acc_cur)} iter: {i}')
                    if is_train or i == len(dl) -1:
                        logs = {
                            f'{descriptor} loss': float(np.mean(loss_cur)),
                            f'{descriptor} accuracy': float(np.mean(acc_cur)),
                        }
                        for j in range(4):
                            logs[f'sigmoid w{j}'] =float(round(torch.sigmoid(self.net.middle_layer[j]).item(),3))
                        wandb.log(logs,step=self.step)


            if is_train:
                self.scheduler.step()
            acc = np.mean(acc_cur)
            self.logger.info(f'Epoch {descriptor} loss: {np.mean(loss_cur)} {descriptor} acc: {acc}')
            return acc

    def train_brain_hemorrhage_detection(self):
        since = time.time()
        best_acc = 0.0
        num_epochs = self.settings.num_epochs

        self.logger.info('starts training brain hemorrhage detection')

        for epoch in range(self.start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            epoch_start_time = time.time()
            self.trainval(is_train=True, dl=self.train_loader)
            epoch_acc_val = self.trainval(is_train=False, dl=self.val_loader)
            epoch_time_elapsed = time.time() - epoch_start_time
            self.logger.info('epoch complete in {:.0f}m {:.0f}s'.format(
                epoch_time_elapsed // 60, epoch_time_elapsed % 60))
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                torch.save({'unet': self.net.state_dict(), 'encoder': self.net.encoder.state_dict()},
                           os.path.join(self.settings.checkpoint_dir, 'unet_best_val_dice.pt'))
            state_dict = {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "sched_state_dict": self.scheduler.state_dict(),
                "model":self.net.state_dict(),
                "run_id": wandb.run_id,
                "step": self.step,
            }
            torch.save(state_dict,os.path.join(self.settings.checkpoint_dir, 'final_ckpt.pt'))

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Acc: {:4f}'.format(best_acc))
