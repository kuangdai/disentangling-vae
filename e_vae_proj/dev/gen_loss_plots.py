import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, help="parent folder of the training results")

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # get loss data
    with open(Path(main_path) / 'results' / 'train_losses.log') as f:
        data = [line.split() for line in f]
        print data