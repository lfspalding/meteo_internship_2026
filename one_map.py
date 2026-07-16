import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
import argparse


def log_function(args):
    print(f"Function called with arguments: {args}")


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


def plot_dataset(data, key):
    # data repair
    data = wrap_data(data, key)
    ax = map_setup()

    # plot data
    cbar_kwargs = {'orientation':'horizontal', 'shrink':1, "pad":.05, 'aspect':60, 'label':'CO2 Level kg/kg'}#, 'format':'%4.2e'
    contour_plot = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, levels=21)
    contour_plot.colorbar.ax.tick_params(labelsize="small")

    # ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    plt.title(data.long_name)
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to data file")
    parser.add_argument("key", help="which param you want to plot")
    parser.add_argument("--plev", type=int, help="pressure level to select")
    parser.add_argument("--time", type=str, help="time to select")

    args = parser.parse_args()
    log_function(args)

    tracerpl_ds = xr.load_dataset(args.file, engine="netcdf4")
    sliced = tracerpl_ds.sel(plev=20, time='1946-08-31T23:52:00.000000000')
    plot_dataset(sliced, args.key)
    