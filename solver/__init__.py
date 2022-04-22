# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from . import config
from .config import get_config, parse_args

from . import solver
from .solver import Solver

from . import dataset
from .dataset import Dataset