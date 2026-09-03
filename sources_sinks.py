import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm, map_setup, remove_seasonal_cycle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm


class CoordinateBox:
    def __init__(self, name, lat0, lat1, lon0, lon1):
        self.name = name
        self.lat0 = lat0
        self.lat1 = lat1
        self.lon0 = lon0
        self.lon1 = lon1

    def __repr__(self):
        return f"CoordinateBox(name={self.name}, lat0={self.lat0}, lat1={self.lat1}, lon0={self.lon0}, lon1={self.lon1})"


def plot_flux(data, box, years):
    data_box = data.sel(
        lat=slice(box.lat0, box.lat1),
        lon=slice(box.lon0, box.lon1),
    )
    area_mean = data_box.mean(dim=['lat', 'lon'])
    deseasonalized = remove_seasonal_cycle(area_mean, key="CO2")
    deseason = deseasonalized.diff(dim='time')
    deseason_ppm = convert_ppm(deseason, key=None)
    deseason_ppm.plot.line(label="Area Flux")

    #Baseline
    world_mean = data.mean(dim=['lat', 'lon'])
    world_deseasonalized = remove_seasonal_cycle(world_mean, key="CO2")
    world_deseason = world_deseasonalized.diff(dim='time')
    world_deseason_ppm = convert_ppm(world_deseason, key=None)
    world_deseason_ppm.plot.line(label="Global Flux", alpha=0.7)

    plt.title(f'CO2 Diff near surface in {box.name}, deseasonalized, years {years[0]}-{years[-1]}')
    plt.xlabel('Year')
    plt.ylabel(f'CO2 Change (ppm)')
    plt.grid()
    plt.legend()
    plt.show()


def map_boxes(data, boxes):
    ax = map_setup()
    crs = ccrs.PlateCarree()

    num_boxes = len(boxes)
    colors = cm.get_cmap('tab10', num_boxes)

    # Loop and plot each box with an index to pick its unique color
    for idx, box in enumerate(boxes):
        # Pick the distinct color for this specific box index
        box_color = colors(idx)
        # Get bounds
        lon_min, lon_max = min(box.lon0, box.lon1), max(box.lon0, box.lon1)
        lat_min, lat_max = min(box.lat0, box.lat1), max(box.lat0, box.lat1)
        
        # Define the 4 corners of the box, closing the loop back at the start
        lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        
        # Plot the path. Cartopy will automatically curve the lines to match the Robinson projection
        ax.plot(
            lons, lats, 
            color=box_color, 
            linewidth=2, 
            transform=crs
        )
        
        # Optional: Add text label
        ax.text(
            lon_min + 1, lat_min + 1, box.name,
            transform=crs, 
            color=box_color, 
            weight='bold'
        )

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
    years = np.arange(1850, 1890, 1, dtype=int)
    layers = np.arange(0, 2, 1, dtype=int)
    boxes = []
    boxes.append(CoordinateBox("Amazon", 5, -20, 280, 310))
    boxes.append(CoordinateBox("Congo Basin", 4, -6, 9, 20))
    boxes.append(CoordinateBox("NA Boreal Forest", 70, 50, 220, 290))
    boxes.append(CoordinateBox("Eurasia Boreal Forest", 70, 50, 20, 120))
    boxes.append(CoordinateBox("South Pole", -80, -90, 0, 360))
    boxes.append(CoordinateBox("North Pole", 90, 80, 0, 360))
    boxes.append(CoordinateBox("E Eq Pacific", 10, -10, 200, 270))
    boxes.append(CoordinateBox("W Eq Pacific", 10, -10, 140, 200))
    boxes.append(CoordinateBox("SE Indian Ocean", -20, -45, 90, 115))
    print(years)
    print(layers)
    print(boxes)
    # lev = 2 # surface flux
    data = data.sel(lev=slice(layers[0], layers[-1])).mean(dim='lev')
    data = data.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-12-31'))
    map_boxes(data, boxes)
    for box in boxes:
        plot_flux(data, box, years)
