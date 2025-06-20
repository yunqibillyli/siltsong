import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.linalg import norm
from matplotlib import ticker
from scipy.integrate import quad
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, PowerNorm
from math import pi, sqrt, sin, cos, acos, atan2, exp, floor, hypot



