from helpers import log_args, remove_seasonal_cycle, remove_seasonal_multi, file_flat, file_zec
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

    # baseline_smooth = baseline["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")
    deseasonalized_base = remove_seasonal_cycle(baseline, key="CO2")
    deseasonalized_plev = remove_seasonal_cycle(plev_data, key="CO2")

    # 4. Difference the smoothed data to kill the massive global trend
    baseline_final = deseasonalized_base.diff(dim="time")
    plev_final = deseasonalized_plev.diff(dim="time")

    # 5. Extract the clean arrays for the CCF function
    y1_prime = plev_final.values
    y2_prime = baseline_final.values

    # 6. Recalculate lags based on the new length
    data_len = len(y1_prime)
    lags_plev = signal.correlation_lags(data_len, len(y2_prime))
    ccf_plev = ccf_values(y1_prime, y2_prime)

    return lags_plev, ccf_plev, data_len, base


def ccf_table(path, base_plev, start, end, zec=False):
    # 1. Fetch your Xarray DataArray and convert Pa to hPa
    da_lag, da_r = ccf_table_to_xarray(path, base_plev, start, end)
    da_lag = da_lag.assign_coords(plev=da_lag.plev / 100.0)
    da_r = da_r.assign_coords(plev=da_r.plev / 100.0)

    # 2. Setup a standard figure canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plt.rcParams.update({"font.size": 12})
    fig.suptitle(f"Time Period: {start} to {end}", fontsize=14, fontweight="bold")
    if (zec):
        fig.suptitle(f"Time Period: {start} to {end}, ZEC", fontsize=14, fontweight="bold")

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

    global_smooth = remove_seasonal_multi(data, key="CO2")
    global_base = global_smooth.sel(plev=base_plev)
    global_targets = global_smooth.sel(plev=plevs)

    # 3. Update the 2D grid shape to use latitude: shape (plev, lat)
    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.empty(grid_shape)
    max_r_grid = np.empty(grid_shape)

    print("Available time steps in slice:", len(global_smooth.time.sel(time=slice(start, end))))
    print("Non-NaN base points:", int((~global_base.isnull()).sum()))

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


def ccf_plot_seasonal(path, base_plev, start, end, season=None):
    da_lag_winter, _ = ccf_seasonal(path, base_plev, start, end, season="Winter")
    da_lag_summer, _ = ccf_seasonal(path, base_plev, start, end, season="Summer")

    # Convert pressure scales from Pa to hPa for plotting consistency
    da_lag_winter = da_lag_winter.assign_coords(plev=da_lag_winter.plev / 100.0)
    da_lag_summer = da_lag_summer.assign_coords(plev=da_lag_summer.plev / 100.0)

    # 2. Subtract them to find the delta
    da_delta_lag = da_lag_winter - da_lag_summer

    # 3. Plot the Difference Map
    fig, ax = plt.subplots(figsize=(10, 7))

    da_delta_lag.plot(
        ax=ax,
        x="lat",
        y="plev",
        cmap="RdBu_r", 
        vmin=-6,       
        vmax=6,
        robust=True,
        cbar_kwargs={"label": "Lag Difference (Winter minus Summer Months)"}
    )

    # 4. Standard vertical atmospheric formatting
    ax.set_yscale("log")
    ax.set_ylim(float(da_delta_lag.plev.values[0]), float(da_delta_lag.plev.values[-1]))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_title("Seasonal Transport Lag ($\Delta$ Lag)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Latitude")

    plt.show()


def remove_long_term_trend_only(data, key=None):
    """
    Removes ONLY the multi-decade background growth trend, 
    leaving the seasonal cycles perfectly intact to act as our transport tracer.
    """
    da = data[key] if key else data
    da = da.sortby("time")
    
    # Use a 5-year (60 month) rolling mean to capture the long-term background rise
    trend = da.rolling(time=60, center=True, min_periods=1).mean()
    
    # Subtract the trend to get a clean seasonal anomaly signal around a zero-baseline
    return da - trend


def ccf_seasonal(path, base_plev, start, end, season=None):
    data = xr.open_dataset(path, engine="netcdf4")

    # 1. Spatial preprocessing
    data = data.sel(time=slice(start, end)).mean(dim="lon")
    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    lats = data.lat.values

    # 2. CRITICAL CHANGE: Keep the seasonal cycles, only drop the multi-decadal trend
    global_seasonal_signals = remove_long_term_trend_only(data, key="CO2")
    global_base = global_seasonal_signals.sel(plev=base_plev)
    global_targets = global_seasonal_signals.sel(plev=plevs)

    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.empty(grid_shape)
    max_r_grid = np.empty(grid_shape)

    years = np.unique(global_seasonal_signals.time.dt.year.values)

    for lat_idx, lat in enumerate(lats):
        base_smooth = global_base.sel(lat=lat)
        
        for plev_idx, plev in enumerate(plevs):
            plev_smooth = global_targets.isel(plev=plev_idx).sel(lat=lat)

            base_align, plev_align = xr.align(base_smooth, plev_smooth, join="inner")
            
            # Use raw anomalies instead of derivatives so we track actual peaks/troughs
            ccf_list = []
            lags = None

            for year in years:
                try:
                    if season == "Winter":
                        # Isolate the winter baseline impulse, track it forward 12 months
                        y_base = base_align.sel(time=slice(f"{year}-12-01", f"{year+1}-02-28"))
                        y_plev = plev_align.sel(time=slice(f"{year}-12-01", f"{year+1}-11-30"))
                    elif season == "Summer":
                        # Isolate the summer baseline impulse, track it forward 12 months
                        y_base = base_align.sel(time=slice(f"{year}-06-01", f"{year}-08-31"))
                        y_plev = plev_align.sel(time=slice(f"{year}-06-01", f"{year+1}-05-31"))
                    else:
                        y_base = base_align.sel(time=f"{year}")
                        y_plev = plev_align.sel(time=f"{year}")

                    b_vals = y_base.dropna(dim="time").values
                    p_vals = y_plev.dropna(dim="time").values

                    if len(b_vals) >= 3 and len(p_vals) >= 3:
                        lags = signal.correlation_lags(len(p_vals), len(b_vals))
                        
                        # Standard Pearson correlation profile across the lags
                        # Ensures proper normalization even with unequal array lengths
                        ccf_year = [np.corrcoef(p_vals[i:i+len(b_vals)], b_vals)[0, 1] 
                                    for i in range(len(p_vals) - len(b_vals) + 1)]
                        
                        # Match the length formatting of correlation_lags for forward steps
                        ccf_forward = np.zeros_like(lags, dtype=float)
                        valid_len = min(len(ccf_year), len(ccf_forward[lags >= 0]))
                        ccf_forward[lags >= 0][:valid_len] = ccf_year[:valid_len]
                        
                        ccf_list.append(ccf_forward)
                except (KeyError, IndexError):
                    continue

            if len(ccf_list) == 0 or lags is None:
                max_xcor_grid[plev_idx, lat_idx] = np.nan
                max_r_grid[plev_idx, lat_idx] = np.nan
                continue

            # Average profiles across the historical era
            mean_ccf = np.nanmean(ccf_list, axis=0)

            non_negative_mask = lags >= 0
            valid_lags = lags[non_negative_mask]
            valid_ccf = mean_ccf[non_negative_mask]
            
            max_idx = np.argmax(valid_ccf)
            max_correlation_value = valid_ccf[max_idx]
            max_r_grid[plev_idx, lat_idx] = max_correlation_value
            
            # Since seasonal cycles provide strong, coherent signals, we can use a firm threshold
            threshold = 0.40 if plev > 20000 else 0.35

            if max_correlation_value < threshold or np.isnan(max_correlation_value):
                max_xcor_grid[plev_idx, lat_idx] = np.nan
            else:
                max_xcor_grid[plev_idx, lat_idx] = valid_lags[max_idx]

    da_lag = xr.DataArray(max_xcor_grid, dims=["plev", "lat"], coords={"plev": plevs, "lat": lats})
    da_r = xr.DataArray(max_r_grid, dims=["plev", "lat"], coords={"plev": plevs, "lat": lats})
    return da_lag, da_r




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    # parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = file_flat
    if (args.zec):
        path = file_zec
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
    ccf_table(path, 8.50e+04, "1900-01", "1950-12", args.zec)
