import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def density_calculation(temp):
    # converts temperature to density
    # parameter:
        # @temp: single value or array of temperatures to be transformed
    densities = 1000 * (1 - ((temp + 288.9414) * (temp - 3.9863)**2) / (508929.2 * (temp + 68.12963)))
    return densities

def lake_energy_calculation(temps, densities, depth_areas):
    # calculate the total energy of the lake for every timestep
    # sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    # then multiply by the specific heat of water 
    dz = 0.5 # thickness for each layer, hardcoded for now
    cw = 4186 # specific heat of water
    depth_areas = torch.reshape(depth_areas, (-1, 1))
    energy = torch.sum(depth_areas * temps * densities * dz * cw, axis=1)
    return energy

def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    # given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
    time = 86400 #seconds per day
    energy_deltas = (energies[:,1:] - energies[:,:-1]) / (time * surface_area)
    return energy_deltas

def calculate_vapour_pressure_saturated(temp):
    # returns in miilibars
    # Converted pow function to exp function workaround pytorch not having autograd implemented for pow
    exponent = (9.28603523 - (2332.37885 / (temp + 273.15))) * np.log(10)
    return torch.exp(exponent)

def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)


def calculate_wind_speed_10m(ws, ref_height = 2.):
    # from GLM code glm_surface.c
    c_z0 = torch.tensor(0.001) #default roughness
    return ws * (torch.log(10.0 / c_z0) / torch.log(ref_height / c_z0))


def calculate_air_density(air_temp, rh):
    # returns air density in kg / m^3
    # equation from page 13 GLM/GLEON paper(et al Hipsey)
    # Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    # atmospheric pressure
    p = 1013. #mb

    # water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)

    # water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure / (p - vapPressure)
    return (1.0 / c_gas * (1 + r)/(1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100

def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 22 in GLM/GLEON paper(et al Hipsey)
    # GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    # calculate air density 
    rho_a = calculate_air_density(air_temp, rel_hum)

    # specific heat capacity of air in J/(kg*C)
    c_a = 1005.

    # bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    # wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    return -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 23 in GLM/GLEON paper(et al Hipsey)
    # GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    # air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    # bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    # latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    # wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
    # 
    # ratio of molecular weight of water to that of dry air
    omega = 0.622

    # air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)

def calculate_energy_fluxes(phys, surf_temps, combine_days):    
    e_s = 0.985 # emissivity of water, given by Jordan
    alpha_sw = 0.07 # shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 # longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 # Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,2] + (phys[1:,2] - phys[:-1,2]) / 2
    R_lw_arr = phys[:-1,3] + (phys[1:,3] - phys[:-1,3]) / 2
    R_lw_out_arr = e_s * sigma * (torch.pow(surf_temps[:] + 273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:] - R_lw_out_arr[:-1]) / 2

    air_temp = phys[:-1,4] 
    air_temp2 = phys[1:,4]
    rel_hum = phys[:-1,5]
    rel_hum2 = phys[1:,5]
    ws = phys[:-1, 6]
    ws2 = phys[1:,6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2) / 2
    H = (H + H2) / 2
    fluxes = (R_sw_arr[:-1] * (1-alpha_sw) + R_lw_arr[:-1] * (1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])
    return fluxes


def energy_fluxes_calculation(phys, surf_temps):
    e_s = 0.985 # emissivity of water, given by Jordan
    alpha_sw = 0.07 # shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 # longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 # Stefan-Baltzmann constant
    
    R_sw_arr = phys[:, :-1, 2] + (phys[:, 1:, 2] - phys[:, :-1, 2]) / 2
    R_lw_arr = phys[:, :-1, 3] + (phys[:, 1:, 3] - phys[:, :-1, 3]) / 2
    R_lw_out_arr = e_s * sigma * (torch.pow(surf_temps[:] + 273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:,:-1] + (R_lw_out_arr[:, 1:] - R_lw_out_arr[:, :-1]) / 2
    
    air_temp = phys[:, :-1, 4] 
    air_temp2 = phys[:, 1:, 4]
    rel_hum = phys[:, :-1,5]
    rel_hum2 = phys[:, 1:,5]
    ws = phys[:, :-1, 6]
    ws2 = phys[:, 1:,6]
    t_s = surf_temps[:, :-1]
    t_s2 = surf_temps[:, 1:]
    
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2) / 2
    H = (H + H2) / 2
    fluxes = (R_sw_arr[:,:-1] * (1-alpha_sw) + \
              R_lw_arr[:,:-1] * (1-alpha_lw) - \
              R_lw_out_arr[:,:-1] + E[:,:-1] + H[:, :-1])
    return fluxes


def EC_loss(preds, phys, depth_areas, n_depths, ec_threshold):
    
    densities = density_calculation(preds)
    lake_energies = lake_energy_calculation(preds, densities, depth_areas=depth_areas)
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, None, depth_areas[0])

    lake_energy_deltas = lake_energy_deltas[:,1:]

    surf_phys = phys[:, 0, :, :]
    surf_pred = preds[:, 0, :]

    lake_energy_fluxes = energy_fluxes_calculation(surf_phys, surf_pred)
    diff_vec = torch.abs(lake_energy_deltas - lake_energy_fluxes) 

    tmp_mask = 1 - phys[:, 0, 1:-1, 9] 
    tmp_loss = torch.mean(diff_vec * tmp_mask, axis = 1)


    ec_threshold = 20
    diff_per_set = torch.clamp(tmp_loss - ec_threshold, min=0, max=999999)
    diff_loss = torch.mean(diff_per_set)
    return diff_loss



class LakeLoss:
    def __init__(self,
                 elam = 0.005, ## loss weight
                 n_depths = None, 
                 depth_areas = None, 
                 ec_threshold = None, 
                 depth_loss = False, 
                 ec_loss = False,
                 device = 'cpu'):
        self.n_depths = n_depths
        self.depth_loss = depth_loss
        self.ec_loss =ec_loss
        self.depth_areas = depth_areas
        self.ec_threshold = ec_threshold
        self.elam = elam
        
        if self.ec_loss:
            if depth_areas is None:
                self.depth_areas = torch.Tensor([
                39865825,38308175,38308175,35178625,35178625,33403850,31530150,31530150,30154150,30154150,29022000,
                29022000,28063625,28063625,27501875,26744500,26744500,26084050,26084050,25310550,24685650,24685650,
                23789125,23789125,22829450,22829450,21563875,21563875,20081675,18989925,18989925,17240525,17240525,
                15659325,14100275,14100275,12271400,12271400,9962525,9962525,7777250,7777250,5956775,4039800,4039800,
                2560125,2560125,820925,820925,216125]).to(device)
                self.n_depths = 50
                self.ec_threshold = 24
    
    def __call__(self, pred, label, mask, phy):
        
        rmse_loss_val = self.weighted_rmse_loss(pred, label, mask)
        if self.ec_loss:
            ec_loss_val = EC_loss(pred, phy, self.depth_areas, self.n_depths, self.ec_threshold)
            total_loss = rmse_loss_val + self.elam * ec_loss_val
        else:
            total_loss = rmse_loss_val
            
        return total_loss
        
    
    def weighted_rmse_loss(self, input, target, weight):
        # defined weighted rmse loss
        # used in model training
        # weight means mask
        return torch.sqrt(torch.sum(weight * (input - target) ** 2) / torch.sum(weight))
