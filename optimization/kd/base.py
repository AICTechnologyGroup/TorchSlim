import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy
import os


class KnowledgeDistillation:

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.temp = temp
        self.distil_weight = distil_weight
        self.loss_fn = loss_fn


        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)


    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        raise NotImplementedError


    def get_parameters(self):

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        pass

    def train_epoch(self):
        raise NotImplementedError()

    def eval_epoch(self):
        raise NotImplementedError()
        
