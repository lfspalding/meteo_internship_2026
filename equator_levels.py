import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import numpy as np

from helpers import convert_ppm


def log_function(args):
    print(f"Function called with arguments: {args}")

def profile_plot(path, years, zec=False):
    plt.rcParams["figure.figsize"]=(10,6)

    cmap = plt.colormaps['plasma']
    norm = matplotlib.colors.Normalize(vmin=1850, vmax=1949)
    year_colors = cmap(norm(years))

    # Create a mappable object using your existing cmap and norm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Draw the colorbar on the side of the plot
    cbar = plt.colorbar(sm, ax=plt.gca(), location="top", orientation="horizontal", pad=0.1, aspect=60)
    cbar.set_label("Year")

    for i, year in enumerate(years):
        year = year
        # time = f'{year}-10-31T23:52:00.000000000'
        string_path = f"{path}/HW310_esm-flat10_RESCUE_echam6_tracerpl_{year}.nc"
        if zec:
            string_path = f"{path}/HW312_esm-flat10-zec_RESCUE_echam6_tracerpl_{year}.nc"
        tracerpl_ds = xr.load_dataset(string_path, engine="netcdf4")
        # sliced = tracerpl_ds.sel(time=time, lon=0, method="nearest")
        # t_avg = tracerpl_ds.sel(time=slice()).mean(dim=time)
        # lon_avg = t_avg.sel(lon=slice()).mean(dim=lon)
        # take average over +/- degrees
        eq_avg = tracerpl_ds.sel(lat=slice(20, -20)).mean(dim=["lat", "lon", "time"])
        # eq_avg = ppm.sel(lat=0, method="nearest") 
        ppm = convert_ppm(eq_avg)


        ppm["CO2"].plot(y="plev", label=f'Year {year}', color=year_colors[i])
        # plt.plot(eq_avg["CO2"], label=f'Year {year}')

    plt.xlabel("CO2 Concentration [ppm]") 
    plt.gca().invert_yaxis()
    plt.title(f"Equator (+/- 20 deg lat) CO2 Concentrations over Time, zec={zec}")
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_function(args)

    years = np.arange(1850, 1950, dtype=int)
    # years = [1946]
    profile_plot(args.path, years, args.zec)

