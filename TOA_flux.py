import argparse
from helpers import log_args
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def calculate_toa_co2_flux(ds, co2_var="CO2", omega_var="w"):
    """
    Calculates the upward mass flux of CO2 at the top-most pressure level.
    
    Parameters:
    ds (xr.Dataset): Dataset with dimensions (time, level, lat, lon)
    co2_var (str): Name of CO2 variable in ppm (dry air mole fraction)
    omega_var (str): Name of vertical velocity variable in Pa/s
    """
    # 1. Grab the top-of-atmosphere pressure layer (e.g., 10 hPa or 100 hPa)
    # This selects the minimum value in the 'level' or 'plev' coordinate axis.
    top_level = ds["level"].min().values
    ds_top = ds.sel(level=top_level)
    
    # 2. Constants for atmospheric conversion
    g = 9.81                 # Gravity (m/s^2)
    M_co2 = 44.01            # Molar mass of CO2 (g/mol)
    M_air = 28.97            # Molar mass of dry air (g/mol)
    
    # 3. Convert CO2 from ppm (parts per million) to a Mass Mixing Ratio (kg_co2 / kg_air)
    # ppm to mole fraction is / 1e6; then multiply by the ratio of molecular weights
    q_co2 = (ds_top[co2_var] / 1e6) * (M_co2 / M_air)
    
    # 4. Extract Omega (vertical pressure velocity in Pa/s)
    omega = ds_top[omega_var]
    
    # 5. Compute Flux (kg of CO2 per square meter per second)
    # A negative omega means upward wind vector, resulting in a positive leaving flux
    co2_flux_density = - (1.0 / g) * q_co2 * omega
    
    # 6. (Optional) Integrate over space to get total global rate (kg/s leaving the top)
    # Requires weighting grid cells by surface area due to converging latitude lines
    weights = np.cos(np.deg2rad(ds_top.lat))
    weights.name = "weights"
    
    # Approximate Earth radius (meters) to find area of grid cells
    R = 6371000 
    dlon = np.deg2rad(np.gradient(ds_top.lon.values)[0])
    dlat = np.deg2rad(np.gradient(ds_top.lat.values)[0])
    grid_area = (R**2) * dlat * dlon * weights
    
    total_mass_rate = (co2_flux_density * grid_area).sum(dim=["lat", "lon"])
    
    # Bundle results into an Xarray Dataset
    flux_ds = xr.Dataset({
        "co2_flux_density": co2_flux_density,  # Units: kg * m^-2 * s^-1
        "total_mass_rate_leaving": total_mass_rate # Units: kg / s
    })
    
    return flux_ds

# AI generated, significantly modified
def calculate_toa_flux_pure_mass(ds, co2_var="co2"):
    """
    Calculates the rate of CO2 crossing the top boundary using a column mass balance.
    
    Parameters:
    ds (xr.Dataset): Dataset with dimensions (time, plev, lat, lon)
                     where 'plev' is pressure in Pascals (Pa).
    co2_var (str): Name of CO2 variable, already in kg/kg (mass mixing ratio).
    """
    # 1. Sort levels from top (lowest pressure) to surface (highest pressure)
    ds = ds.sortby("plev", ascending=True)
    plev_vals = ds.plev.values 
    
    # 2. Calculate the pressure thickness (dp) of each vertical layer boundary
    dp = np.diff(plev_vals) 
    dp = np.append(dp, dp[-1]) # Pad the last element to keep arrays aligned
    
    # Map the numpy array back to an Xarray DataArray matching 'plev'
    dp_da = xr.DataArray(dp, coords={"plev": ds.plev}, dims=["plev"])
    
    # 3. Calculate total integrated CO2 column mass (kg / m^2)
    g = 9.81
    air_mass_per_layer = dp_da / g
    co2_mass_per_layer = ds[co2_var] * air_mass_per_layer
    total_column_co2_mass = co2_mass_per_layer.sum(dim="plev")
    world_co2_mass = total_column_co2_mass.sum(dim=["lat", "lon"])
    
    # 4. Calculate change over time (dM/dt) in seconds
    time_diff_seconds = ds.time.diff(dim="time").dt.total_seconds()
    dM = world_co2_mass.diff(dim="time")
    dM_dt = dM / time_diff_seconds # Units: kg * s^-1
    
    # 5. Mass Balance: Leaving flux equals the negative accumulation rate
    co2_flux = -dM_dt
    flux_per_month = co2_flux * 86400 * 30.44  # Units: kg / m² / month

    aligned_time = ds.time[1:]
    co2_flux = co2_flux.assign_coords(time=aligned_time)
    flux_per_month = flux_per_month.assign_coords(time=aligned_time)
    total_column_co2_mass = total_column_co2_mass.assign_coords(time=ds.time)
    
    return xr.Dataset({
        "total_column_co2_mass": total_column_co2_mass,  # Units: kg/m^2
        "co2_flux": co2_flux,      # Units: kg / s
        "flux_per_month": flux_per_month,  # Units: kg / month
        "world_co2_mass": world_co2_mass  # Units: kg
    })


def plot_toa_flux(flux_ds):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()  
    
    world_flux = flux_ds["flux_per_month"]
    world_flux.plot.line(ax=ax1, label="Global Total CO2 Flux (kg/month)", color='lightblue')

    # nh_flux = flux_ds["flux_per_month"].sel(lat=slice(90, 0)).sum(dim=["lat", "lon"])
    # nh_flux.plot.line(label="Northern Hemisphere CO2 Top Leaving Flux (g/m²/month)", color='green') 

    # sh_flux = flux_ds["flux_per_month"].sel(lat=slice(0, -90)).sum(dim=["lat", "lon"])
    # sh_flux.plot.line(label="Southern Hemisphere CO2 Top Leaving Flux (g/m²/month)", color='red')

    world_mass = flux_ds["world_co2_mass"]
    world_mass.plot.line(ax=ax2, label="Global CO2 Mass (kg)", color='blue')
    
    ax1.set_ylim(-10000, 10000)
    ax2.set_ylim(85000, 105000)

    plt.title("Top of Atmosphere CO2 Flux and Total Mass Over Time")
    plt.xlabel("Time")
    plt.ylabel("Flux / Mass")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracerpl_1850-1949.nc'
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA YET")
        # path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'
    print(path)
    data = xr.load_dataset(path, engine="netcdf4")
    years = np.arange(1850, 1870, 1, dtype=int)
    print(years)
    data = data.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-12-31'))

    flux_data = calculate_toa_flux_pure_mass(data, co2_var="CO2")
    print(flux_data.flux_per_month)
    plot_toa_flux(flux_data)
