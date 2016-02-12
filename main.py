__author__ = 'mashers'


import os
import threading

from raspi import raspi_main

if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    raspi = False

    raspi_main.run_raspi_modules(raspi, pwd)

