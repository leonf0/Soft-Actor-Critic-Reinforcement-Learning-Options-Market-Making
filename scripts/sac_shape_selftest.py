import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.sac import sac_shape_self_test

_ = sac_shape_self_test()