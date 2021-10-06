import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time

class Bubble:

    def __init__(self, q, omega, Disp, FormFac, mu):
        self.q=q
        self.omega=omega
        self.Disp=Disp
        self.FormFac=FormFac
        self.mu=mu
        