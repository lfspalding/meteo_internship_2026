import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm, file_flat
import matplotlib.pyplot as plt


def co2_vs_time(data, years, lat1, lat2, lon1, lon2):
    data = data.sel(
        lat=slice(lat1, lat2),
        lon=slice(lon1, lon2),
        time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"),
    )
    data = data.sel(plev=slice(100000, 1000))
    data = data.mean(dim=["lat", "lon"])
    data_ppm = convert_ppm(data)
    data_ppm["plev"] = data_ppm["plev"] / 100
    data_ppm["CO2"].transpose("plev", "time").plot.contourf(
        x="time", y="plev", cmap="viridis", levels=20
    )
    plt.gca().invert_yaxis()

    plt.title("CO2 Vertical Profile Over Time")
    plt.ylabel("Pressure Level (hPa)")
    plt.xlabel("Time")
    plt.title(
        f"CO2 Concentration Over Time, lat {lat1} to {lat2}, lon {lon1} to"
        f" {lon2}, years {years[0]}-{years[-1]}"
    )
    plt.yscale("log")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = file_flat
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA")
    print(path)

    data = xr.load_dataset(path, engine="netcdf4")
    
    years = np.arange(1875, 1900, 1, dtype=int)
    # Amazon
    lat1, lat2 = 5, -20 
    lon1, lon2 = 280, 310
    print(years)
    co2_vs_time(data, years, lat1, lat2, lon1, lon2)
