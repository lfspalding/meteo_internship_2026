import calendar

import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
import pandas as pd
import numpy as np
from cartopy.util import add_cyclic_point
from statsmodels.tsa.seasonal import STL


def convert_ppm(data, key="CO2"):
    conversion_factor = (44.01 / 28.97) * 1e6
    if key: 
        data[key] = data[key] * conversion_factor
        data[key].attrs["units"] = "ppm"
    else:
        data = data * conversion_factor
        data.attrs["units"] = "ppm"
    # Convert CO2 from ppm to kg/kg
    # 1 ppm = 1e-6 (dimensionless)
    # Molar mass CO2 ≈ 44.01 g/mol, Molar mass air ≈ 28.97 g/mol
    return data


def convert_hpa(data):
    # Convert pressure from Pa to hPa
    data["plev"] = data["plev"] / 100.0
    data["plev"].attrs["units"] = "hPa"
    return data


def remove_seasonal_cycle(data, key=None):

    # 1. Sort by time to ensure calculations happen chronologically
    data_sorted = data.sortby("time")
    
    # 2. Convert to pandas Series (required by statsmodels STL)
    if key:
        series = data_sorted[key].to_series()
    else:
        series = data_sorted.to_series()
    
    # 3. Fit STL to perfectly isolate the smooth trend line
    # period=12 for monthly data; robust=True handles any outliers/missing steps
    stl = STL(series, period=12, robust=True).fit()
    deseasonalized_trend = stl.trend
    anomalies_sorted = xr.DataArray.from_series(deseasonalized_trend)
    datetime_fixed = pd.to_datetime(anomalies_sorted.time.values) + pd.offsets.MonthEnd(0)
    anomalies_sorted = anomalies_sorted.assign_coords(time=datetime_fixed)
    return anomalies_sorted


def remove_seasonal_multi(data, key=None):
    # 1. Extract the DataArray if a key is provided
    da = data[key] if key else data
    
    # 2. Sort by time to ensure calculations happen chronologically
    da = da.sortby("time")
    
    # 3. Define a helper function to run STL on a 1D numpy array
    def _stl_trend_1d(array_1d):
        # Handle entirely empty grid cells (like subsurface topography masks)
        if np.isnan(array_1d).all() or len(array_1d) < 12:
            return np.full_like(array_1d, np.nan)
        
        stl = STL(array_1d, period=12, robust=True).fit()
        return stl.trend

    # 4. Use apply_ufunc to map the 1D function across the 'time' axis automatically
    deseasonalized = xr.apply_ufunc(
        _stl_trend_1d,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,  # Automatically loops over plev and lat under the hood
        output_dtypes=[da.dtype]
    )
    
    # 5. Restore clean MonthEnd timestamps to match your original formatting
    datetime_fixed = pd.to_datetime(deseasonalized.time.values) + pd.offsets.MonthEnd(0)
    return deseasonalized.assign_coords(time=datetime_fixed)



def log_args(args):
    print(f"Function called with arguments: {args}")


def get_month(year, num):
    months = [f'{year}-01-31T23:52:00.000000000', 
              f'{year}-02-28T23:52:00.000000000', 
              f'{year}-03-31T23:52:00.000000000', 
              f'{year}-04-30T23:52:00.000000000', 
              f'{year}-05-31T23:52:00.000000000', 
              f'{year}-06-30T23:52:00.000000000', 
              f'{year}-07-31T23:52:00.000000000', 
              f'{year}-08-31T23:52:00.000000000', 
              f'{year}-09-30T23:52:00.000000000',
              f'{year}-10-31T23:52:00.000000000',
              f'{year}-11-30T23:52:00.000000000', 
              f'{year}-12-31T23:52:00.000000000']
    if calendar.isleap(year):
        months[1] = f'{year}-02-29T23:52:00.000000000'
    return months[num - 1]


def wrap_data(data, key=False):
    if key:
        da = data[key]
    else:
        da = data
    lon_idx = list(da.dims).index('lon')
    cyclic_data, cyclic_lons = add_cyclic_point(da.values, coord=da.lon.values, axis=lon_idx)
    new_coords = dict(da.coords)
    new_coords['lon'] = cyclic_lons
    
    cyclic_da = xr.DataArray(
        cyclic_data,
        coords=new_coords,
        dims=da.dims,
        attrs=da.attrs
    )
    return cyclic_da


def map_setup():
    projection = ccrs.Robinson()
    crs = ccrs.PlateCarree()
    plt.figure(dpi=150)
    ax = plt.axes(projection=projection, frameon=True)
    gl = ax.gridlines(crs=crs, draw_labels=True,
                  linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size" : 7}
    gl.ylabel_style = {"size" : 7}
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    # ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)
    return ax