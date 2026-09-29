import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import numpy as np

from helpers import convert_ppm, get_month, convert_hpa, log_args


def max_CO2_plot(path, years, months, plevs, zec=False):
    plt.rcParams["figure.figsize"] = (15, 6)
    fig, ax = plt.subplots()

    colors = plt.cm.turbo(np.linspace(0, 1, len(plevs)))

    for i, plev in enumerate(plevs):

        peak_lats = []
        max_co2 = []

        for year in years:
            string_path = f"{path}/HW310_esm-flat10_RESCUE_echam6_tracerpl_{year}.nc"
            if zec:
                string_path = f"{path}/HW312_esm-flat10-zec_RESCUE_echam6_tracerpl_{year}.nc"
            
            tracerpl_ds = xr.load_dataset(string_path, engine="netcdf4")
            sel_plev = tracerpl_ds.sel(plev=plev).mean(dim="lon") #, lat=slice(45, -45)

            # peak_lats.extend(sel_plev["CO2"].idxmax(dim="lat").values)
            # max_co2.extend(sel_plev["CO2"].max(dim="lat").values)
            peak_lats.extend(sel_plev["CO2"].idxmax(dim="lat").values.flatten())
            max_co2.extend(sel_plev["CO2"].max(dim="lat").values.flatten())
        
        # Your verified time indexing line
        time_values = years[0] + np.arange(len(peak_lats)) / 12.0        
        # Scale marker sizes directly using raw CO2 variations
        marker_sizes = (np.array(max_co2) - np.min(max_co2) + 1) * 10
        
        # Plot unified line and scatter elements simultaneously 
        ax.plot(time_values, peak_lats, zorder=plev, color=colors[i], label=f'Plev {plev} Pa')
        # ax.scatter(time_values, peak_lats, s=marker_sizes, color=colors[i], edgecolors='black', alpha=0.8, zorder=2)
        # 
        peak_lats_da = xr.DataArray(peak_lats)
        
        # Calculate a centered 12-month rolling average to isolate the yearly trend
        yearly_avg_lats = peak_lats_da.rolling(dim_0=12, center=True, min_periods=1).mean().values   
        ax.plot(
            time_values, 
            yearly_avg_lats, 
            zorder=plev + 1000,          # Ensures it sits cleanly on top of the dots
            color=colors[i], 
            linewidth=1.5,              
            linestyle="--", 
            label=f'Plev {plev} Pa (Yearly Avg)',
            alpha=0.7
        )     
    
    # Clean up X-axis labels to tick on the start of each year
    ax.set_xticks(years)
    
    ax.set_ylabel("Latitude ($^\circ$N)")
    ax.set_title("Location of Maximum Atmospheric $CO_2$ Concentration Over Time", fontweight='bold')
    # ax.set_ylim(-45, 45)
    ax.set_xlim(years[0] - 0.10, years[-1] + 1.10)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.grid(True, which='both', axis='x', linestyle="--", color="gray", alpha=0.6)
    ax.grid(True, axis='y', linestyle="--", color="gray", alpha=0.6)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5, zorder=0)
    
    # plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/'
    if (args.zec):
        path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'

    years = np.arange(1940, 1945, dtype=int)
    print(years)
    months = np.arange(1, 13, dtype=int)
    print(months)
    plevs = [1.00e+04, 7.00e+03, 5.00e+03, 3.00e+03]
    print(plevs)
    max_CO2_plot(path, years, months, plevs, args.zec)
