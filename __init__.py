#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from control import NetworksControl


def main():
    mode = 1  # 0 - train  1 - run
    network_name = 'conv_s0'
    path_to_save = os.getcwd() + '/saved/' + network_name
    path_to_src = os.getcwd() + '/src/'

    if mode==0:
        num_epochs=20

        nc=NetworksControl(path_to_src, path_to_save)
        nc.train(num_epochs)

    if mode==1:
        path_to_input_image =  os.getcwd() + '/test/s_120_5_0_60.jpg'

        nc = NetworksControl(path_to_src, path_to_save)
        output=nc.run(path_to_input_image)
        print('Letter in image - ' + output[0])

if __name__ == "__main__":
    main()





