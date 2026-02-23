## imports

from argopy import DataFetcher
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import cmocean

## constants
### bounding boxes
NAGulfStream_BB = [45, -90, 25, -45] # top-left, bottom-right
Kuroshio_BB = [50.5, 114.0, 0, 180]

### other
START = '2025-03-01'
END = '2025-03-21'

### OBJECTIVE MAPPING
xcor_km, ycor_km = 300, 300

R = 6378.1 #Radius of earth

# Add a small amount of noise to represent the instrument and measurement error.
# This also helps to ensure E is invertible
err = 0.05

## functions

def fetch_ARGO_data(bounding_box, 
                    min_depth=0, max_depth = 200,
                    start_date=START, end_date=END):

    lon_min = bounding_box[1] # -80
    lon_max = bounding_box[3] # -45

    lat_min = bounding_box[2] # 30
    lat_max = bounding_box[0] # 45

    ArgoSet = DataFetcher(mode='expert', parallel=True, progress=True).region(
        [lon_min, lon_max,
        lat_min, lat_max,
        min_depth, max_depth,
        start_date, end_date]
    ).load()

    ds_profiles = ArgoSet.data.argo.point2profile()

    return ds_profiles

def PROJECTION_CHOICES(bounding_box):

    lon_min = bounding_box[1] # -80
    lon_max = bounding_box[3] # -45

    lat_min = bounding_box[2] # 30
    lat_max = bounding_box[0] # 45

    mid_lon = (lon_max+lon_min)/2
    mid_lat = (lat_max+lat_min)/2

    return {
        'az_eq': ccrs.AzimuthalEquidistant(
        central_longitude=mid_lon,   # midpoint of lon range
        central_latitude=mid_lat,    # midpoint of lat range
    ),  'igh': ccrs.InterruptedGoodeHomolosine(
        central_longitude=mid_lon
    ),  'albers_ee': ccrs.AlbersEqualArea(
        central_longitude=mid_lon,   # midpoint of lon range
        central_latitude=mid_lat,     # midpoint of lat range
        standard_parallels=(lat_min, lat_max)
    ),  'mercator': ccrs.Mercator(
        central_longitude=mid_lon,
        min_latitude=lat_min,
        max_latitude=lat_max,
        latitude_true_scale=mid_lat
    ),  'lambert': ccrs.LambertConformal(
        central_longitude=mid_lon,   # midpoint of lon range
        central_latitude=mid_lat,     # midpoint of lat range
        standard_parallels=(lat_min, lat_max)
    ), 'pc': ccrs.PlateCarree(
        central_longitude=mid_lon
    )
    }

def format_geoaxes(ax, xlabel_style = {'size': 9}, ylabel_style = {'size': 9}):
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = xlabel_style
    gl.ylabel_style = ylabel_style

def calculate_objective_map(ds, xcor_km=xcor_km, ycor_km=ycor_km, R=R, err=err, nx=100):

    temperature = ds.where(ds.TEMP_QC == 1).TEMP
    lat = ds.LATITUDE
    lon = ds.LONGITUDE

    temp_mean = temperature.mean(dim='N_LEVELS')

    data = temp_mean.values
    x = lon.values
    y = lat.values

    aspect_ratio = len(y) / len(x)
    ny = int(aspect_ratio * nx)

    xg = np.linspace(x.min()-0.1, x.max()+0.1, nx)
    yg = np.linspace(y.min()-0.1, y.max()+0.1, ny)

    # de-mean the data
    dm = np.mean(data)
    dat = data - dm

    # Data-data covariance matrix E
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    Y1, Y2 = np.meshgrid(y, y, indexing='ij')
    lat_rad = np.deg2rad(Y1)
    dx_km = (X1 - X2) * (R * np.pi / 180) * np.cos(lat_rad)
    dy_km = (Y1 - Y2) * (R * np.pi / 180)
    E = np.exp(-dx_km**2 / xcor_km**2 - dy_km**2 / ycor_km**2)
    E += err * np.eye(len(dat))

    # Data-grid covariance matrix C
    C = np.zeros((len(dat), nx * ny))
    Xg, Yg = np.meshgrid(xg, yg, indexing='ij')
    Xg_flat = Xg.flatten(order='F')
    Yg_flat = Yg.flatten(order='F')
    for n in range(len(dat)):
        lat_rad = np.deg2rad((y[n] + Yg_flat) / 2)  # shape (N_grid,)
        dx_km = (Xg_flat - x[n]) * (R * np.pi / 180) * np.cos(lat_rad).T
        dy_km = (Yg_flat - y[n]) * (R * np.pi / 180)
        C[n, :] = np.exp(-dx_km**2 / xcor_km**2 - dy_km**2 / ycor_km**2)

    # Make the matrix of functions for a planar fit:
    F = np.zeros((len(dat), 3))
    F[0:len(x), 0] = x
    F[0:len(y), 1] = y
    F[0:len(x), 2] = 1

    # Make the matrix of the functions evaluated at the grid points:
    grid_points = np.column_stack([Xg_flat, Yg_flat])

    f = np.vstack([grid_points[:,0],grid_points[:,1],np.ones(grid_points.shape[0])])

    # Calculate the gain A
    EF = np.linalg.solve(E, F) # This is a more numerically stable version of calculating E^{-1}F
    W = np.linalg.solve(F.T @ EF, EF.T).T # E^{-1}F(F^TEF)^{-1}
    EC = np.linalg.solve(E, C) # E^{-1}C
    A = W @ f + (np.eye(len(dat)) - W @ F.T) @ EC

    # Calculate the objective map
    map = (A.T @ dat).reshape(nx, ny, order='F')
    map_plot = map + dm # Add back in the mean

    # Calcualte the error grid
    ergrid = np.zeros(nx * ny)
    EA = E @ A #
    for n in range(nx * ny):
        ergrid[n] = A[:, n].T @ (EA[:, n] - 2 * C[:, n]) + 1 # E and C are already normalised by the variance <T'^2> so this is the normalised MSE
    ergrid = ergrid.reshape(nx, ny, order='F')

    return map_plot, ergrid, xg, yg