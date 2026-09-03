# TODO
# look at surface layer and see where CO2 is added
# maybe look at next layer too

import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm, remove_seasonal_cycle
import matplotlib.pyplot as plt


def surface_flux(data, years):
    area_mean = data.mean(dim=['lat', 'lon'])
    diffs = area_mean.diff(dim='time')
    diffs_ppm = convert_ppm(diffs["CO2"], key=None)

    yearly_diffs = diffs.resample(time="6MS").mean()
    yearly_diffs_ppm = convert_ppm(yearly_diffs["CO2"], key=None)

    deseasonalized = remove_seasonal_cycle(area_mean, key="CO2")
    deseason = deseasonalized.diff(dim='time')
    deseason_ppm = convert_ppm(deseason, key=None)

    rolling_twelve = diffs_ppm.rolling(time=12, center=True).mean().dropna(dim="time")

    return diffs_ppm, yearly_diffs_ppm, deseason_ppm, rolling_twelve, years


def plot_fluxes(monthly, yearly, deseasonalized, rolling, years):
    monthly.plot.line(label="Monthly Flux")
    yearly.plot.line(label="Yearly Flux")
    deseasonalized.plot.line(label="Deseasonalized Flux")
    rolling.plot.line(label="Rolling Twelve Month Mean")

    plt.title(f'CO2 Diff near surface, deseasonalized, years {years[0]}-{years[-1]}')
    plt.xlabel('Year')
    plt.ylabel(f'CO2 Change (ppm)')
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    # parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracer_1850-1949.nc'
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA YET")
        # path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'
    print(path)
    data = xr.load_dataset(path, engine="netcdf4")
    years = np.arange(1870, 1890, 1, dtype=int)
    layers = np.arange(0, 5, 1, dtype=int)
    print(years)
    print(layers)
    # lev = 2 # surface flux
    data = data.sel(lev=slice(layers[0], layers[-1])).mean(dim='lev')
    data = data.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-12-31'))
    plot_fluxes(surface_flux(data))

