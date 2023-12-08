import numpy as np
import torch
import logging
import os
import sys
import h5py
import csv
import time
import random
import json
from datetime import datetime
from torch.utils.data import Dataset
from utils import int16_to_float32

# For ESC dataset
class ESC_Dataset(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.dataset = self.dataset[self.config.esc_fold]
        else:
            temp = []
            for i in range(len(self.dataset)):
                if i != config.esc_fold:
                    temp += list(self.dataset[i]) 
            self.dataset = temp           
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))


    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": np.concatenate((self.dataset[p]["waveform"],self.dataset[p]["waveform"])),
            "real_len": len(self.dataset[p]["waveform"]) * 2,
            "target": self.dataset[p]["target"]
        }
        return data_dict

    def __len__(self):
        return self.total_size

