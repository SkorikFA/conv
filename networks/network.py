#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self,outputs_count):
        super(Convolutional, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        )

        self.drop_out = nn.Dropout(0.5)
        self.line1 = nn.Linear(4864, 512)
        self.tanh = nn.Tanh()
        self.line2 = nn.Linear(512, outputs_count)

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.drop_out(x)
        x = self.line1(x)
        x = self.tanh(x)
        x = self.line2(x)

        return x