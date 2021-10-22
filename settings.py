import os
import time
from shutil import copy
from Logger import LoggerHandler


"""
the following code is to define the model settings from a dictionary of settings which include the following keys:
-- dataset_settings
-- pre_processing_settings
-- dataloader_settings
-- augmentations_settings
-- compilation_settings
-- metrics_settings
-- output_settings
-- architecture_settings
-- training_settings
"""

class Settings(object):
    """
    this class is to define the DADE project settings
    """
    def __init__(self, settings_dict, write_logger,exp,results_dir,resume):

        # pre processing settings
        self.clipping = settings_dict['pre_processing_settings']['clip']
        if self.clipping:
            self.min_clip_val = settings_dict['pre_processing_settings']['min_val']
            self.max_clip_val = settings_dict['pre_processing_settings']['max_val']
        else:
            self.min_clip_val = None
            self.max_clip_val = None

        # compilation settings
        self.optimizer = settings_dict['compilation_settings']['optimizer']
        self.gamma_decay = settings_dict['compilation_settings']['gamma_decay']
        self.lr_decay_step_size = settings_dict['compilation_settings']['lr_decay_step_size']
        self.lr_decay_policy = settings_dict['compilation_settings']['lr_decay_policy']
        self.initial_learning_rate = settings_dict['compilation_settings']['initial_learning_rate']
        self.weights_init = settings_dict['compilation_settings']['weights_init']
        self.weight_decay = settings_dict['compilation_settings']['weight_decay']

        if self.optimizer == 'adam':
            self.beta_1 = settings_dict['compilation_settings']['beta_1']
            self.beta_2 = settings_dict['compilation_settings']['beta_2']
        # output_settings
        self.simulation_folder = os.path.join(results_dir,exp)
        self.checkpoint_dir = os.path.join(self.simulation_folder, 'checkpoint')
        if not os.path.exists(self.simulation_folder):
            os.mkdir(self.simulation_folder)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if resume:
            if not os.path.exists(os.path.join(self.checkpoint_dir, 'final_ckpt.pt')):
                raise Exception("can't resume without final ckpt")
        self.resume = resume
        self.exp = exp


        # architecture settings
        self.encoder_name = settings_dict['architecture_settings']['encoder_name']
        self.encoder_depth = settings_dict['architecture_settings']['encoder_depth']
        self.encoder_weights = settings_dict['architecture_settings']['encoder_weights']
        self.decoder_use_batchnorm = settings_dict['architecture_settings']['decoder_use_batchnorm']
        self.decoder_channels = settings_dict['architecture_settings']['decoder_channels']
        self.n_channels = settings_dict['architecture_settings']['in_channels']
        self.classes = settings_dict['architecture_settings']['n_classes']
        self.input_size = settings_dict['architecture_settings']['input_size']
        self.net_type = settings_dict['architecture_settings']['net_type']
        self.lr_for_middle_layer = settings_dict['architecture_settings']['lr_for_middle_layer']
        self.layer_wise = settings_dict['architecture_settings']['layer_wise']
        self.rand = settings_dict['architecture_settings']['rand']
        self.only_middle = settings_dict['architecture_settings']['only_middle']

        # training settings
        self.train_model = settings_dict['training_settings']['train_model']
        self.batch_size = settings_dict['training_settings']['batch_size']
        self.num_epochs = settings_dict['training_settings']['num_epochs']


        # logger settings
        self.save_image_iter = settings_dict['logger_settings']['save_image_iter']
        self.image_display_iter = settings_dict['logger_settings']['image_display_iter']
        self.display_size = settings_dict['logger_settings']['display_size']
        self.log_iter = settings_dict['logger_settings']['log_iter']
        self.snapshot_save_iter = settings_dict['logger_settings']['snapshot_save_iter']
        self.save_loss_to_log = settings_dict['logger_settings']['save_loss_to_log']
        if write_logger:
            if self.train_model:
                self.logger_name = 'logger_train'
                self.output_logs = os.path.join(self.simulation_folder, 'results')
            else:
                self.logger_name = 'logger'
                self.output_logs = os.path.join(self.simulation_folder, 'inference')
            if not os.path.exists(self.output_logs):
                os.mkdir(self.output_logs)
            self.output_logs += os.sep
            self.log_message = ''
            self.logger_handler = LoggerHandler(self)
            self.logger_handler.start(exp)
            self.logger = self.logger_handler.logger
            self.logger.debug(self.log_message)

