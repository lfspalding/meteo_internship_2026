from helpers import log_args, remove_seasonal_cycle, remove_seasonal_multi, file_flat, file_zec
import numpy as np
from scipy import signal
import xarray as xr
import matplotlib.pyplot as plt
import argparse


def ccf_seasonal(data, base_plev, start, end, season=None, max_lag_months=12):
    data = data.sel(time=slice(start, end)).mean(dim="lon")
    
    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    lats = data.lat.values

    # Linear Detrending
    def detrend_dim(da, dim):
        val = da.values
        def _detrend(y):
            if np.isnan(y).any():
                return np.full_like(y, np.nan)
            return signal.detrend(y)
        detrended = np.apply_along_axis(_detrend, da.get_axis_num(dim), val)
        return xr.DataArray(detrended, coords=da.coords, dims=da.dims)

    global_seasonal_signals = detrend_dim(data['CO2'], dim='time')
    global_base = global_seasonal_signals.sel(plev=base_plev)
    global_targets = global_seasonal_signals.sel(plev=plevs)

    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.full(grid_shape, np.nan)

    months = global_base.time.dt.month
    if season == "Winter":
        seasonal_mask = (months == 12) | (months == 1) | (months == 2)
    elif season == "Summer":
        seasonal_mask = (months == 6) | (months == 7) | (months == 8)
    else:
        seasonal_mask = xr.DataArray(np.ones(len(global_base.time), dtype=bool), coords=[global_base.time])

    for lat_idx, lat in enumerate(lats):
        base_series = global_base.sel(lat=lat)
        
        for plev_idx, plev in enumerate(plevs):
            target_series = global_targets.isel(plev=plev_idx).sel(lat=lat)
            base_align, target_align = xr.align(base_series, target_series, join="inner")
            
            ccf_profile = []
            
            for lag in range(max_lag_months + 1):
                target_shifted = target_align.shift(time=-lag)
                
                b_vals = base_align.where(seasonal_mask, drop=True).values
                p_vals = target_shifted.where(seasonal_mask, drop=True).values
                
                valid_mask = ~np.isnan(b_vals) & ~np.isnan(p_vals)
                b_clean = b_vals[valid_mask]
                p_clean = p_vals[valid_mask]
                
                if len(b_clean) > 10:
                    # FIXED: Added [0, 1] to strictly capture the scalar correlation coefficient
                    r_val = np.corrcoef(b_clean, p_clean)[0, 1]
                    ccf_profile.append(r_val)
                else:
                    ccf_profile.append(np.nan)
            
            if not np.all(np.isnan(ccf_profile)):
                max_idx = np.nanargmax(ccf_profile)
                max_correlation = ccf_profile[max_idx]
                
                # PHYSICS FIX: If the maximum correlation is incredibly weak (< 0.25), 
                # it means the seasonal cycle has faded out. Leave it as NaN (white space).
                if max_correlation >= 0.25:
                    max_xcor_grid[plev_idx, lat_idx] = max_idx

    da_lag = xr.DataArray(max_xcor_grid, dims=["plev", "lat"], coords={"plev": plevs, "lat": lats})
    return da_lag

def ccf_plot_seasonal(data, base_plev, start, end, zec=False):
    # FIXED: Explicitly set max_lag_months=6 to isolate the true physical transport path
    da_lag_winter = ccf_seasonal(data, base_plev, start, end, season="Winter", max_lag_months=6)
    da_lag_summer = ccf_seasonal(data, base_plev, start, end, season="Summer", max_lag_months=6)

    da_lag_winter = da_lag_winter.assign_coords(plev=da_lag_winter.plev / 100.0)
    da_lag_summer = da_lag_summer.assign_coords(plev=da_lag_summer.plev / 100.0)

    da_delta_lag = da_lag_winter - da_lag_summer
    da_delta_lag = da_delta_lag.sortby("plev", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Bounded levels for a 6-month maximum search window
    levels = np.arange(-6, 7, 1)
    contour = ax.contourf(
        da_delta_lag.lat,
        da_delta_lag.plev,
        da_delta_lag.values,
        levels=levels,
        cmap="RdBu_r",
        extend="both"
    )
    
    cbar = fig.colorbar(contour, ax=ax, orientation="vertical", pad=0.05)
    cbar.set_label("Lag Difference (Winter minus Summer Months)")

    ax.set_yscale("log")
    ax.invert_yaxis() 
    ax.set_ylim(float(da_delta_lag.plev.values[-1]), float(da_delta_lag.plev.values[0]))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_title("Seasonal Transport Lag ($\Delta$ Lag)", fontsize=12, fontweight="bold")
    if zec:
        ax.set_title("Seasonal Transport Lag ($\Delta$ Lag), ZEC", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Latitude")

    plt.show()


def ccf_plot_seperate(data, base_plev, start, end, zec=False):
    # Fetch seasonal lag datasets (6-month max search window)
    da_lag_winter = ccf_seasonal(
        data, base_plev, start, end, season="Winter", max_lag_months=12
    )
    da_lag_summer = ccf_seasonal(
        data, base_plev, start, end, season="Summer", max_lag_months=12
    )

    # Convert Pa to hPa
    da_lag_winter = da_lag_winter.assign_coords(plev=da_lag_winter.plev / 100.0)
    da_lag_summer = da_lag_summer.assign_coords(plev=da_lag_summer.plev / 100.0)

    # Sort pressure levels consistently
    da_lag_winter = da_lag_winter.sortby("plev", ascending=True)
    da_lag_summer = da_lag_summer.sortby("plev", ascending=True)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Absolute lag range bounds (0 to 6 months)
    levels = np.arange(0, 13, 1)
    cmap = "viridis"  # Sequential colormap preferred for positive absolute lag values

    # --- Plot Winter ---
    contour_winter = ax1.contourf(
        da_lag_winter.lat,
        da_lag_winter.plev,
        da_lag_winter.values,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax1.set_title("Winter Transport Lag", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- Plot Summer ---
    contour_summer = ax2.contourf(
        da_lag_summer.lat,
        da_lag_summer.plev,
        da_lag_summer.values,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax2.set_title("Summer Transport Lag", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latitude")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- Shared Y-Axis Formatting (Log scale + Inverted Pressure) ---
    # Both axes share configurations, but settings limits on ax1 dictates both due to sharey=True
    ax1.invert_yaxis()
    ax1.set_ylim(
        float(da_lag_winter.plev.values[-1]),
        float(da_lag_winter.plev.values[0]),
    )

    # --- Unified Colorbar ---
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_summer, cax=cbar_ax) 
    cbar.set_label("Lag (Months)")
    fig.suptitle(f"Global CO2 Transport Lag Analysis {start}-{end}", fontsize=16, fontweight="bold", y=0.98)
    if (zec):
        fig.suptitle(f"Global CO2 Transport Lag Analysis {start}-{end}, ZEC", fontsize=16, fontweight="bold", y=0.98)

    plt.show()


def ccf_yearly(data, base_plev, start, end, season=None, max_lag_years=5):
    data = data.sel(time=slice(start, end)).mean(dim="lon")

    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    lats = data.lat.values

    # Linear Detrending
    def detrend_dim(da, dim):
        val = da.values

        def _detrend(y):
            if np.isnan(y).any():
                return np.full_like(y, np.nan)
            return signal.detrend(y)

        detrended = np.apply_along_axis(_detrend, da.get_axis_num(dim), val)
        return xr.DataArray(detrended, coords=da.coords, dims=da.dims)

    global_seasonal_signals = detrend_dim(data["CO2"], dim="time")
    global_base = global_seasonal_signals.sel(plev=base_plev)
    global_targets = global_seasonal_signals.sel(plev=plevs)

    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.full(grid_shape, np.nan)

    # 1. Determine steps per year based on data frequency (e.g., 12 for monthly data)
    time_delta = data.time.diff(dim="time").values
    # Check if data is monthly by looking at the median days between steps (~30 days)
    is_monthly = np.median(time_delta).astype("timedelta64[D]").astype(int) in range(
        27, 32
    )
    steps_per_year = 12 if is_monthly else 1  # Default to 1 if already annual

    # Seasonal Mask Setup
    months = global_base.time.dt.month
    if season == "Winter":
        seasonal_mask = (months == 12) | (months == 1) | (months == 2)
    elif season == "Summer":
        seasonal_mask = (months == 6) | (months == 7) | (months == 8)
    else:
        seasonal_mask = xr.DataArray(
            np.ones(len(global_base.time), dtype=bool), coords=[global_base.time]
        )

    for lat_idx, lat in enumerate(lats):
        base_series = global_base.sel(lat=lat)

        for plev_idx, plev in enumerate(plevs):
            target_series = global_targets.isel(plev=plev_idx).sel(lat=lat)
            base_align, target_align = xr.align(
                base_series, target_series, join="inner"
            )

            ccf_profile = []

            # 2. Iterate through yearly lags
            for lag_years in range(max_lag_years + 1):
                # Calculate the exact index shift required for a full calendar year
                idx_shift = lag_years * steps_per_year
                target_shifted = target_align.shift(time=-idx_shift)

                # 3. Apply the seasonal mask AFTER the calendar-year shift
                b_vals = base_align.where(seasonal_mask, drop=True).values
                p_vals = target_shifted.where(seasonal_mask, drop=True).values

                valid_mask = ~np.isnan(b_vals) & ~np.isnan(p_vals)
                b_clean = b_vals[valid_mask]
                p_clean = p_vals[valid_mask]

                if len(b_clean) > 10:
                    r_val = np.corrcoef(b_clean, p_clean)[0, 1]
                    ccf_profile.append(r_val)
                else:
                    ccf_profile.append(np.nan)

            if not np.all(np.isnan(ccf_profile)):
                max_idx = np.nanargmax(ccf_profile)
                max_correlation = ccf_profile[max_idx]

                if max_correlation >= 0.25:
                    # Output is now directly mapped to the year index (0, 1, 2... years)
                    max_xcor_grid[plev_idx, lat_idx] = max_idx

    da_lag = xr.DataArray(
        max_xcor_grid, dims=["plev", "lat"], coords={"plev": plevs, "lat": lats}
    )
    return da_lag


def ccf_plot_yearly(data, base_plev, start, end, max_lag_years=5, zec=False):
    # 1. CHANGED: Call the yearly function instead of the seasonal one, and pass max_lag_years
    da_lag_winter = ccf_yearly(
        data, base_plev, start, end, season="Winter", max_lag_years=max_lag_years
    )
    da_lag_summer = ccf_yearly(
        data, base_plev, start, end, season="Summer", max_lag_years=max_lag_years
    )

    # Convert Pa to hPa
    da_lag_winter = da_lag_winter.assign_coords(plev=da_lag_winter.plev / 100.0)
    da_lag_summer = da_lag_summer.assign_coords(plev=da_lag_summer.plev / 100.0)

    # Sort pressure levels consistently
    da_lag_winter = da_lag_winter.sortby("plev", ascending=True)
    da_lag_summer = da_lag_summer.sortby("plev", ascending=True)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # 2. CHANGED: Match contour levels to your maximum year lag instead of 12 months
    levels = np.arange(0, max_lag_years + 1, 1)
    cmap = "viridis" 

    # --- Plot Winter ---
    contour_winter = ax1.contourf(
        da_lag_winter.lat,
        da_lag_winter.plev,
        da_lag_winter.values,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax1.set_title("Winter Transport Lag", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- Plot Summer ---
    contour_summer = ax2.contourf(
        da_lag_summer.lat,
        da_lag_summer.plev,
        da_lag_summer.values,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax2.set_title("Summer Transport Lag", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latitude")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- Shared Y-Axis Formatting (Log scale + Inverted Pressure) ---
    ax1.invert_yaxis()
    ax1.set_ylim(
        float(da_lag_winter.plev.values[-1]),
        float(da_lag_winter.plev.values[0]),
    )

    # --- Unified Colorbar ---
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_summer, cax=cbar_ax) 
    
    # 3. CHANGED: Update the colorbar label from "Months" to "Years"
    cbar.set_label("Lag (Years)")
    fig.suptitle(f"Global CO2 Transport Lag Analysis {start}-{end}", fontsize=16, fontweight="bold", y=0.98)
    if (zec):
        fig.suptitle(f"Global CO2 Transport Lag Analysis {start}-{end}, ZEC", fontsize=16, fontweight="bold", y=0.98)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = file_flat
    if (args.zec):
        path = file_zec
    print(path)

    data = xr.open_dataset(path, engine="netcdf4")

    ccf_plot_seperate(data, 8.50e+04, "1860-01", "1950-12", zec=args.zec)
    ccf_plot_yearly(data, 8.50e+04, "1860-01", "1950-12", zec=args.zec)

