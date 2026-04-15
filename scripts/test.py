import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))

from eval_mri import main


if __name__ == "__main__":
    if "--split" not in sys.argv:
        sys.argv.extend(["--split", "test"])
    main()
