import calendar

import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
from cartopy.util import add_cyclic_point


def convert_ppm(data):
    # Convert CO2 from ppm to kg/kg
    # 1 ppm = 1e-6 (dimensionless)
    # Molar mass CO2 ≈ 44.01 g/mol, Molar mass air ≈ 28.97 g/mol
    conversion_factor = (44.01 / 28.97) * 1e-6
    data["CO2"] = data["CO2"] * conversion_factor
    data["CO2"].attrs["units"] = "ppm"
    return data


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
              f'{year}-09-31T23:52:00.000000000', 
              f'{year}-11-31T23:52:00.000000000', 
              f'{year}-12-31T23:52:00.000000000']
    if calendar.isleap(year):
        months[1] = f'{year}-02-29T23:52:00.000000000'
    return months[num - 1]


def wrap_data(data, key):
    da = data[key]
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