#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,random
import cv2
import torch
import torch.nn as nn
from networks.network import Convolutional

class NetworksControl:
    def __init__(self,path_to_src, path_to_save_dir):
        self.path_to_letters_src=path_to_src
        self.path_to_save=path_to_save_dir
        # self.device = torch.device('cuda:0')
        self.device = torch.device('cpu')
        self.letters = self.get_letters_by_path(self.path_to_letters_src)

    def train(self, num_epochs):
        net = None
        if os.path.exists(self.path_to_save) == False:
            net = Convolutional(len(self.letters))
        else:
            net = torch.load(self.path_to_save)
        # net.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

        for epoch in range(num_epochs):
            inputs = self.get_inputs(self.path_to_letters_src)
            total_step = len(inputs)

            iterations_count = len(inputs)
            for i in range(iterations_count):
                lt = self.letters[i]

                file_name = self.get_next_file(inputs[i])
                if file_name == None:
                    inputs[i] = os.listdir(self.path_to_letters_src + lt + '/' + inputs[i])
                    continue

                input = self.tensor_by_path_create(self.path_to_letters_src + lt + '/' + file_name, self.device)
                input = input.float()
                y = net(input)

                output = self.output_tensor_create_CrossEntropyLoss(lt, self.letters, self.device).float()
                loss = criterion(y, output.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        torch.save(net, self.path_to_save)
        print('network saved')

    def run(self, path_to_letter):
        net = torch.load(self.path_to_save)

        input = self.tensor_by_path_create(path_to_letter, self.device)
        input = input.float()
        output = net(input)

        val_max=-1
        index_max=0
        for i in range(output.size()[1]):
            value=output[0][i].item()
            if value>val_max:
                val_max=value
                index_max=i

        return [self.letters[index_max],val_max]

    def get_letters_by_path(self, path_to_letters_src):
        files = os.listdir(path_to_letters_src)
        files.sort()
        letters = ''.join(files)

        return letters

    def get_inputs(self, path_to_letters_src):
        inputs=[]
        files = os.listdir(path_to_letters_src)
        files.sort()

        for dir_name in files:
            images_names = os.listdir(path_to_letters_src+dir_name)
            inputs.append(images_names)

        return inputs

    def get_next_file(self,inputs):
        file_name=None
        if len(inputs)>0:
            i=random.randint(0,len(inputs)-1)
            file_name=inputs[i]
            inputs.pop(i)

        return file_name

    def output_tensor_create_CrossEntropyLoss(self,lt,letters,device):
        flag_ok=False
        output=[]
        for i in range(len(letters)):
            lt_current=letters[i]
            if lt_current==lt:
                output.append(i)
                flag_ok = True
                break

        if flag_ok==False:
            return torch.tensor([], device=device)
        else:
            return torch.tensor(output, device=device)

    def tensor_by_path_create(self, path, device):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        (h, w) = image.shape
        input = []

        for y in range(h):
            xline = []
            for x in range(w):
                color = image[y, x] / 255
                xline.append(color)
            input.append(xline)

        input=torch.tensor([[input]], device=device, requires_grad=True)

        return input