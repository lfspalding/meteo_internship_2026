import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse

from helpers import convert_ppm, wrap_data, log_args


def build_subplots(data, plevels, key, time):
    fig, axes = plt.subplots(nrows=2, ncols=(len(plevels)+1)//2, figsize=(12, 8), subplot_kw={'projection': ccrs.Robinson()})
    axes = axes.flatten()
    min_val = data[key].sel(plev=slice(plevels[-1], plevels[0])).min().values
    max_val = data[key].sel(plev=slice(plevels[-1], plevels[0])).max().values    

    for i, plevel in enumerate(plevels):
        # Select the data for the current pressure level
        data_at_plevel = data.sel(plev=plevel)
        data_at_plevel = wrap_data(data_at_plevel, key)

        # Plot the data on the corresponding axis
        ax = axes[i]
        ax.set_global()
        ax.coastlines()
        cbar_kwargs = {'orientation':'horizontal', 'shrink':1, "pad":.05, 'aspect':60, 'label':f'CO2 Level {data[key].attrs.get("units", "unknown")}'}
        contour_plot = data_at_plevel.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, 
                                                    levels=29, vmin=min_val, vmax=max_val)
        contour_plot.colorbar.ax.tick_params(labelsize="x-small")
        ax.set_title(f'Pressure Level: {plevel} Pa')

    plt.suptitle(f'CO2 Levels for {time}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to data file")
    parser.add_argument("year", help="year of data file")
    parser.add_argument("--zec", action="store_true", help="use zec data")
    # parser.add_argument("key", help="which param you want to plot")
    # parser.add_argument("--plev", type=int, help="pressure level to select")
    # parser.add_argument("--time", type=str, help="time to select")

    args = parser.parse_args()
    log_args(args)

    string_path = f"{args.path}/HW310_esm-flat10_RESCUE_echam6_tracerpl_{args.year}.nc"
    if args.zec:
        string_path = f"{args.path}/HW312_esm-flat10-zec_RESCUE_echam6_tracerpl_{args.year}.nc"
    plevels = [500, 700, 1000, 2000, 3000, 5000]
    time = f'{args.year}-08-31T23:52:00.000000000'

    tracerpl_ds = xr.load_dataset(string_path, engine="netcdf4")
    sliced = tracerpl_ds.sel(time=time)
    ppm = convert_ppm(sliced)
    build_subplots(ppm, plevels, 'CO2', time)
