import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse

from helpers import convert_ppm, log_args
from related_phenomena import enso_index


def yearly_difference(data, years, plev):
    """
    Calculate the yearly difference of CO2 levels for a given pressure level across specified years.

    Parameters:
    data (xarray.Dataset): The dataset containing CO2 levels.
    years (list): List of years to consider for the difference calculation.
    plev (float): The pressure level to select.

    Returns:
    xarray.DataArray: The yearly difference of CO2 levels.
    """
    world_mean = data.mean(dim=['lat', 'lon'])
    diffs = world_mean.diff(dim='time')
    yearly_diffs = diffs.groupby("time.year").mean()
    yearly_diffs_ppm = convert_ppm(yearly_diffs)
    yearly_diffs_ppm["CO2"].plot.line()
    # x, y = yearly_diffs_ppm["year"], yearly_diffs_ppm["CO2"]
    # plt.plot(x, np.polyval(np.polyfit(x, y, 1), x), "r--", label="Trend")
    plt.title(f'Yearly CO2 Diff at {plev/100} hPa, years {years[0]}-{years[-1]}, -30-90° lat')
    plt.xlabel('Year')
    plt.ylabel(f'CO2 Change (ppm)')
    plt.grid()
    plt.show()
    
    return yearly_diffs


def monthly_difference(data, years, plev):
    """
    Calculate the monthly difference of CO2 levels for a given pressure level across specified years.

    Parameters:
    data (xarray.Dataset): The dataset containing CO2 levels.
    years (list): List of years to consider for the difference calculation.
    plev (float): The pressure level to select.

    Returns:
    xarray.DataArray: The monthly difference of CO2 levels.
    """
    world_mean = data.mean(dim=['lat', 'lon'])
    diffs = world_mean.diff(dim='time')
    monthly_diffs = diffs.groupby("time.month").mean()
    monthly_diffs = convert_ppm(monthly_diffs)
    monthly_diffs["CO2"].plot.line()
    plt.title(f'Monthly CO2 Diff at {plev/100} hPa, years {years[0]}-{years[-1]}, -30-90° lat')
    plt.xlabel('Month')
    plt.ylabel(f'CO2 Change (ppm)')
    plt.grid()
    plt.show()
    
    return monthly_diffs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    # parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracerpl_1850-1949.nc'
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA YET")
        # path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'
    print(path)
    data = xr.load_dataset(path, engine="netcdf4")
    years = np.arange(1850, 1950, 1, dtype=int)
    print(years)
    plev = 8.50e+04

    data = data.sel(plev=plev, lat=slice(-30, -90), time=slice(f'{years[0]}-01-01', f'{years[-1]}-12-31'))

    # yearly_difference(data, years, plev)
    # monthly_difference(data, years, plev)
    enso_index()