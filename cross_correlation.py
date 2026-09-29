from helpers import log_args, remove_seasonal_cycle, file_flat
import numpy as np
from scipy import signal
import xarray as xr
import matplotlib.pyplot as plt
import argparse


def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))  
    c = np.correlate(p, q, 'full')
    return c


def create_ccf_plot(lags, ccf, data_len, ax, ylim, color):
    ax.plot(lags, ccf, color=color, label="correlation")
    ax.axhline(-1.96/np.sqrt(data_len), color='red', label='95% confidence interval')
    ax.axhline(1.96/np.sqrt(data_len), color='red')
    ax.axvline(x = 0, color = 'black', lw = 1)
    ax.axhline(y = 0, color = 'black', lw = 1)
    ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, 
    linestyle='--', label = 'highest +/- correlation')
    ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, 
    linestyle='--')
    # ax.set(ylim = ylim)
    ax.set_ylabel('Correlation Coefficients', weight='bold', fontsize = 12)
    ax.set_xlabel('Time Lag (Months)', weight='bold', fontsize = 12)
    # ax.set_xlim(-max-min, max-min)
    ax.legend()


def just_math_ccf(path, plevs):
    data = xr.load_dataset(path, engine="netcdf4")
    start_time = '1900-01-31T23:52:00.000000000'
    end_time = '1930-12-31T23:52:00.000000000'
    data = data.sel(time=slice(start_time, end_time))
    for plev in plevs:
        lags_plev, ccf_plev, data_len, base = do_calc(data, plev)
        max_idx = np.argmax(ccf_plev)
        exact_lag = lags_plev[max_idx]
        max_corr = ccf_plev[max_idx]
        print(f"{plev/100} hPa: Max corr {max_corr:.3f} at {exact_lag} Months")


def plot_ccf(path, plev, zec=False):
    data = xr.load_dataset(path, engine="netcdf4")

    lags_plev, ccf_plev, data_len, base = do_calc(data, plev)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_title(f'Cross Correlation of CO2 at {plev/100} hPa vs Baseline {base/100} hPa', weight='bold', fontsize = 15)
    create_ccf_plot(lags_plev, ccf_plev, data_len, ax, [-0.75, 0.75], "green")
    
    # some additional analysis 
    max_idx = np.argmax(ccf_plev)
    exact_lag = lags_plev[max_idx]
    max_corr = ccf_plev[max_idx]
    print(f"The maximum correlation of {max_corr:.3f} occurs at a lag of {exact_lag} timesteps.")
    
    plt.show()


def do_calc(data, plev):
    # compare to baseline
    base = 9.25e+04
    
    # 1. Extract the raw slices
    baseline = data.sel(plev=base, lat=slice(30, -30)).mean(dim=["lon", "lat"])
    plev_data = data.sel(plev=plev, lat=slice(30, -30)).mean(dim=["lon", "lat"])

    # 2. Apply a 12-month rolling mean to completely smooth out the seasonal cycle
    baseline_smooth = baseline["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")
    plev_smooth = plev_data["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")

    # 3. Align the time coordinates after dropping NaN edges
    baseline_smooth, plev_smooth = xr.align(baseline_smooth, plev_smooth)

    # 4. Difference the smoothed data to kill the massive global trend
    baseline_final = baseline_smooth.diff(dim="time")
    plev_final = plev_smooth.diff(dim="time")

    # 5. Extract the clean arrays for the CCF function
    y1_prime = plev_final.values
    y2_prime = baseline_final.values

    # 6. Recalculate lags based on the new length
    data_len = len(y1_prime)
    lags_plev = signal.correlation_lags(data_len, len(y2_prime))
    ccf_plev = ccf_values(y1_prime, y2_prime)

    return lags_plev, ccf_plev, data_len, base


def ccf_table(path, base_plev, start, end):
    # 1. Fetch your Xarray DataArray and convert Pa to hPa
    da_lag, da_r = ccf_table_to_xarray(path, base_plev, start, end)
    da_lag = da_lag.assign_coords(plev=da_lag.plev / 100.0)
    da_r = da_r.assign_coords(plev=da_r.plev / 100.0)

    # 2. Setup a standard figure canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plt.rcParams.update({"font.size": 12})
    fig.suptitle(f"Time Period: {start} to {end}", fontsize=14, fontweight="bold")

    # 3. Clean, direct 2D plot call using built-in Xarray wrappers
    # First plot: Lag profiles
    da_lag.plot(
        ax=ax1,
        x="lat",
        y="plev",
        vmax=50,
        cmap="inferno_r",
        robust=True,
        cbar_kwargs={"label": "Lag (Months)"}, 
    )
    ax1.set_title(f"Max X-Corr Lag Profile (Base: {base_plev/100.0} hPa)")

    # Second plot: Correlation strength profiles
    da_r.plot(
        ax=ax2,
        x="lat",
        y="plev",
        vmin=0.0,
        vmax=1.0,  # Limits colorbar range to valid correlation bounds
        cmap="viridis",  # Using a distinct cmap helps visually separate the two variables
        robust=True,
        cbar_kwargs={"label": "Pearson R-value"}, 
    )
    ax2.set_title("Peak Correlation Matrix Strength ($R$-value)")

    # 4. Standard structural grid formatting applied to BOTH subplots
    y_start = float(da_lag.plev.values[0])   
    y_end = float(da_lag.plev.values[-1])   
    custom_ticks = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 300, 500, 700]
    valid_ticks = [t for t in custom_ticks if min(y_start, y_end) <= t <= max(y_start, y_end)]

    for ax in [ax1, ax2]:
        # Apply Logarithmic Scale to the Pressure Axis
        ax.set_yscale("log")
        
        # Lock historical vertical view constraints 
        ax.set_ylim(y_start, y_end)
        
        # Set tick locations and map decimal formats nicely
        ax.set_yticks(valid_ticks)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:g}"))
        
        # Labels and background guides
        ax.set_ylabel("Pressure (hPa)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_xlabel("Latitude")

    plt.show()

def ccf_table_to_xarray(path, base_plev, start, end):
    # AI + manual tweaks
    data = xr.load_dataset(path, engine="netcdf4")

    # Take Lat Stripes (zonal mean)
    data = data.sel(time=slice(start, end)).mean(dim="lon")
    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    lats = data.lat.values

    ## TODO will R values get better with local averaging?

    # Pre-compute rolling means globally without dropping NaNs
    global_smooth = data["CO2"].rolling(time=12, center=True).mean()
    global_base = global_smooth.sel(plev=base_plev)
    global_targets = global_smooth.sel(plev=plevs)

    # 3. Update the 2D grid shape to use latitude: shape (plev, lat)
    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.empty(grid_shape)
    max_r_grid = np.empty(grid_shape)

    # 4. Iterate over latitudes and pressure levels
    for lat_idx, lat in enumerate(lats):
        # 1. Isolate the target slices with their original time dimensions intact
        base_smooth = global_base.sel(lat=lat)
        
        for plev_idx, plev in enumerate(plevs):
            plev_smooth = global_targets.isel(plev=plev_idx).sel(lat=lat)

            # 2. Align them first so their time steps perfectly match
            base_align, plev_align = xr.align(base_smooth, plev_smooth, join="inner")

            # 3. Take the time derivative
            base_diff = base_align.diff(dim="time")
            plev_diff = plev_align.diff(dim="time")

            # season_mask = base_diff.time.dt.month.isin([6, 7, 8])
            # base_diff = base_diff.where(season_mask, drop=True)
            # plev_diff = plev_diff.where(season_mask, drop=True)

            # 4. NOW drop any time step where EITHER series has a NaN (e.g., the rolling edges)
            # This keeps the remaining arrays perfectly synchronized in time
            valid_time = ~(base_diff.isnull() | plev_diff.isnull())
            base_values = base_diff.where(valid_time, drop=True).values
            plev_values = plev_diff.where(valid_time, drop=True).values

            # SAFETY CHECK: Prevent np.argmax crash on empty slices or insufficient data
            # Minimum length of 3 required to calculate meaningful standard deviation/diffs
            if len(base_values) < 3 or len(plev_values) < 3:
                max_xcor_grid[plev_idx, lat_idx] = np.nan
                continue

            lags = signal.correlation_lags(len(plev_values), len(base_values))
            ccf = ccf_values(plev_values, base_values)

            non_negative_mask = lags >= 0
            
            # Filter both arrays to only evaluate forward-moving time
            valid_lags = lags[non_negative_mask]
            valid_ccf = ccf[non_negative_mask]
            
            # Find the maximum index within your filtered subset
            max_idx = np.argmax(valid_ccf)
            max_correlation_value = valid_ccf[max_idx]
            max_r_grid[plev_idx, lat_idx] = max_correlation_value
            
            # Earth's atmosphere typically requires R >= 0.3 to be meaningful here
            if plev > 200: 
                threshold = 0.3  # Dynamic baseline for lower atmosphere
            else:
                threshold = 0.25 # Dynamic baseline for upper atmosphere / stratosphere

            if max_correlation_value < threshold:
                max_xcor_grid[plev_idx, lat_idx] = np.nan
            else:
                max_xcor_grid[plev_idx, lat_idx] = valid_lags[max_idx]

    # 5. Build clean, labeled DataArrays for both metrics
    da_lag = xr.DataArray(
        data=max_xcor_grid,
        dims=["plev", "lat"],
        coords={"plev": plevs, "lat": lats},
        name="max_lag",
        attrs={"description": f"Cross-correlation lag relative to base_plev {base_plev}", "units": "months"}
    )
    da_r = xr.DataArray(
        data=max_r_grid,
        dims=["plev", "lat"],
        coords={"plev": plevs, "lat": lats},
        name="max_r",
        attrs={"description": f"Peak Pearson correlation coefficient at the max_lag step", "units": "Pearson R"}
    )
    return da_lag, da_r



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = file_flat
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA YET")
    print(path)
    # years = np.arange(1920, 1946, 5, dtype=int)
    # print(years)
    # months = [9] # np.arange(1, 13, dtype=int)
    # print(months)
    # plot_ccf(path, args.plev, args.zec)
    # plevs = [8.50e+04, 7.75e+04, 7.00e+04, 6.00e+04, 5.00e+04,
    #    4.00e+04, 3.00e+04, 2.50e+04, 2.00e+04, 1.50e+04, 1.00e+04, 7.00e+03,
    #    5.00e+03, 3.00e+03, 2.00e+03, 1.00e+03, 7.00e+02, 5.00e+02, 3.00e+02,
    #    2.00e+02, 1.00e+02, 5.00e+01, 2.00e+01, 1.00e+01]
    # just_math_ccf(path, plevs)
    ccf_table(path, 8.50e+04, "1860-01", "1949-12")
