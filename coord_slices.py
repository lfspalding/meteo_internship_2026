import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import numpy as np

from helpers import convert_ppm, get_month, convert_hpa


def log_function(args):
    print(f"Function called with arguments: {args}")

def profile_plot(path, years, months, plev, zec=False):
    plt.rcParams["figure.figsize"]=(10,6)

    Y, M = np.meshgrid(years, months)

    # 3. Flatten them to create aligned 1D arrays of 36 elements each
    flat_years = Y.flatten()
    flat_months = M.flatten()

    # 4. Sort them chronologically so 1850 (Jan-Dec) comes before 1851 (Jan-Dec)
    sort_indices = np.lexsort((flat_months, flat_years))
    flat_years = flat_years[sort_indices]
    flat_months = flat_months[sort_indices]

    # 5. Convert to 36 continuous fractional year values
    time_values = flat_years + (flat_months - 1) / 12.0

    # 6. Set up the colormap and map the 36 unique colors
    cmap = plt.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=np.min(time_values), vmax=np.max(time_values))
    time_colors = cmap(norm(time_values))

    # Create a mappable object using your existing cmap and norm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Draw the colorbar on the side of the plot
    tick_positions = years
    tick_labels = [str(y) for y in years]
    cbar = plt.colorbar(sm, ax=plt.gca(), location="top", orientation="horizontal", ticks=tick_positions, pad=0.1, aspect=60)
    cbar.set_label("Year")
    cbar.ax.set_xticklabels(tick_labels)

    for i, year in enumerate(years):
        string_path = f"{path}/HW310_esm-flat10_RESCUE_echam6_tracerpl_{year}.nc"
        if zec:
            string_path = f"{path}/HW312_esm-flat10-zec_RESCUE_echam6_tracerpl_{year}.nc"
        
        tracerpl_ds = xr.load_dataset(string_path, engine="netcdf4")
        sel_plev = tracerpl_ds.sel(plev=plev) #, lat=slice(45, -45)

        for m_idx, month in enumerate(months):            
            time = get_month(year, month)
            sel_time = sel_plev.sel(time=time)
        # take average over longitude
            avg_other = sel_time.mean(dim=["lon"])
            ppm = convert_hpa(convert_ppm(avg_other))
        # plot
            ppm["CO2"].plot(y="lat", label=f'{year}:{month}', color=time_colors[i*(len(months))+m_idx])
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
    parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_function(args)

    years = np.arange(1851, 1855, dtype=int)
    months = np.arange(1, 13, 3, dtype=int)
    # years = [1946]
    profile_plot(args.path, years, months, args.plev, args.zec)
