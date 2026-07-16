import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import numpy as np

from helpers import convert_ppm


def log_function(args):
    print(f"Function called with arguments: {args}")

def profile_plot(path, years, plev, zec=False):
    plt.rcParams["figure.figsize"]=(10,6)

    cmap = plt.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=1890, vmax=max(years))
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
        sel_plev = tracerpl_ds.sel(plev=plev)
        # take average over +/- degrees
        avg_other = sel_plev.mean(dim=["lon", "time"])
        ppm = convert_ppm(avg_other)

        ppm["CO2"].plot(y="lat", label=f'Year {year}', color=year_colors[i])
        # plt.plot(eq_avg["CO2"], label=f'Year {year}')

    plt.xlabel("CO2 Concentration [ppm]") 
    # plt.gca().invert_yaxis()
    plt.title(f"Plev {plev} CO2 Concentrations over Time vs Latitude, zec={zec}")
    # plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_function(args)

    years = np.arange(1890, 1930, dtype=int)
    # years = [1946]
    profile_plot(args.path, years, args.plev, args.zec)
