import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))

from train_mri import main


if __name__ == "__main__":
    main()
