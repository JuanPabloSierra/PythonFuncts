#### Importing the libraries needed ####
import numpy.ma as ma
import numpy.ma as ma
import rasterio
import random
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.stats
from netCDF4 import Dataset
from scipy import stats
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

#### This function generates plots for fields from the WRF metgrid files over the entire domain ####

def ScalarPlot(x,y,field, title, minim, maxim, step, colors, reverse=False):
    """
    This function generates the map of a desired field for the whole spatial domain.
   
    Parameters:
    x: (array) array variable where the longitude is located 
    y: (array) array variable where the latitude is located 
    field: (array) array variable where the field is located
    Title: (str) Title for the plot
    minim: (int) minimum value in the color bar
    maxim: (int) maximum value in the color bar
    step: (int) step in the colorbar
    colors: (str) desired palette of colors
    reverse: (bool) if you want revers or not the palette of colors

    Returns:
    Figure of the 2d field
    """
    if reverse==True:
        colors=colors+'_r'
    fig, ax = plt.subplots(figsize=(9,6),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2,linestyle='solid')
    cf=ax.contourf(x,y,field,levels=np.arange(minim,maxim,step),extend='both',cmap=colors,transform=ccrs.PlateCarree())
    plt.colorbar(cf,shrink=0.5)
    plt.title(title)
    return plt.show()

def VectorPlot(x,y,fieldx, fieldy, title, scalar, skip,fieldz=[],colored=False):
    """
    This function generates the map of vectors of a desired fields for the whole spatial domain.
   
    Parameters:
    x: (array) array variable where the longitude is located 
    y: (array) array variable where the latitude is located 
    fieldx: (array) array variable where the component in x is located
    fieldy: (array) array variable where the component in y is located
    fieldz: (array) array variable where the component in z is located (for color map)
    Title: (str) Title for the plot
    scalar: (int) control the size of the vectors (hte higher scalar the smaller the vector)
    skip: (int) control ho many vectors will be skiped in order to get a cleaner plot
    colored: (bool) True for colored map False (default) for not colored map.
    
    Returns:
    Figure of the 2d vector field
    """
    fig, ax = plt.subplots(figsize=(9,6),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2,linestyle='solid')
    if colored==True:
        minim=input('Which minimum value will be used for the colorbar? ')
        minim=int(minim)
        maxim=input('Which maximum value will be used for the colorbar? ')
        maxim=int(maxim)
        step=input('Which step value will be used for the colorbar? ')
        step=int(step)       
        colors=input('Which color palette do you want to use? ')
        cf=ax.contourf(x,y,fieldz,levels=np.arange(minim,maxim,step),extend='both',cmap=colors,transform=ccrs.PlateCarree())
        plt.colorbar(cf,shrink=0.5)    
    ax.quiver(x[::skip,::skip],y[::skip,::skip],fieldx[::skip,::skip],fieldy[::skip,::skip],transform=ccrs.PlateCarree(), scale=scalar,headwidth=4, headlength=6)
    plt.title(title)
    return plt.show()

def FindPointRegIPSL(longitudes,latitudes,acc_delta,Lon,Lat):
    """
    This functions looks for the location i,j of an specific geographical point (longitude, latitude)
    in the RegIPSL grid.

    Parameters:
    longitudes: (array) array variable of the nav_lon_grid_M from RegIPSL.
    latitudes: (array) array variable of the nav_lon_grid_M from RegIPSL.
    acc_delta: (float) corresponds to the accepted difference between the interest point real coordinates and the closer coordinates in the RegIPSL grid.
    Lon,Lat: (float) refers to the longitude/latitude coordinates of the interest point.
    
    Returns:
    Longitud,Latitud: (floats) Correspond to the indexes of the interest point in the RegIPSL grid.
    They should be used in the lon lat arrays like lat[Longitud,Latitud] and lon[Longitud,Latitud].
    
    Help:
    If you get the following error message: "UnboundLocalError: local variable 'Longitud' referenced before assignment"
    Please slightly increase the acc_delt parameter. 

    """
    A=np.abs(longitudes-Lon)
    B=np.abs(latitudes-Lat)
    MenoresA=np.where(A<acc_delta)
    MenoresB=np.where(B<acc_delta)
    # Convertimos la tupla en coordenadas X,Y #
    LonX=[]
    for x in range(MenoresA[0].shape[0]):
        LonX.append([MenoresA[0][x],MenoresA[1][x]])
    LatX=[]
    for x in range(MenoresB[0].shape[0]):
        LatX.append([MenoresB[0][x],MenoresB[1][x]])

    # Evaluamos los lugares donde tanto nuestra latitud como nuestra longitud coinciden #
    for i in range(len(LonX)):
        if LonX[i] in LatX:
           Longitud,Latitud=LonX[i] # Estos son los indices de la longitud y latitud de nuestro punto de interes
    return Longitud,Latitud

# Falta crear una funcion para graficar puntos y para graficar vectores 
