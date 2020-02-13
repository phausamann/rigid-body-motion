"""
Simultaneous head and gaze tracking
===================================
"""
from __future__ import print_function

import xarray as xr

import rigid_body_motion as rbm


##############################################################################
# First, we load the datasets and plot the linear velocity of the head
# tracker in world coordinates.
#
# .. tip::
#
#     In the jupyter notebook version, change the first cell to ``%matplotlib
#     notebook`` in order to get an interactive plot that you can zoom and pan.

# %%
import os
os.chdir('examples')

# %%
gaze = xr.open_dataset('data/gaze.nc')
odometry = xr.open_dataset('data/odometry.nc')

# %%
odometry.linear_velocity.sel(cartesian_axis='x').plot()
