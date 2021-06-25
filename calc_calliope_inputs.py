#############################################################################
# Program : calc_calliope_inouts.py
# Author  : Sarah Sparrow
# Date    : 23/06/2021
# Purpose : Calculate calliope inputs from ERA5 for CPDN data based on scripts in 
#           energy_model_functions from Hannah Bloomfield
#############################################################################

import energy_model_functions_CPDN as energy_model_functions
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import calendar
import matplotlib.ticker as ticker
import pandas as pd

# Define dictionary of countries acronyms - may be possible to lookup from shapefile
countries={'Ireland':'IRL','United Kingdom':'GBR'}

# Define input data directory
ddir='/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_7/cpdn_example_data/remap_ERA5/'

prefix='CPDN'

hub_height=100.

def calliope_inputs_country(country,hub_height):
    # Solar PV
    t2m_data,country_mask,lon,lat,dates = energy_model_functions.load_country_weather_data(country,ddir,'tas_1hrly_mean_a000_2018-01_2018-12.nc','item3236_1hrly_mean','time1',0)
    ssrd_data,country_mask,lon,lat, dates_rad = energy_model_functions.load_country_weather_data(country,ddir,'rsds_1hrly_mean_a000_2018-01_2018-12.nc','field203','time0',0)
    solar_pv_cf = energy_model_functions.solar_PV_model(t2m_data,ssrd_data,country_mask)

    # Demand
    t2m_daily_data,country_mask,lons,lats,dates_daily = energy_model_functions.load_country_weather_data(country,ddir,'tas_1hrly_mean_a000_2018-01_2018-12.nc','item3236_1hrly_mean','time1',24)
    hdd, cdd = energy_model_functions.calc_hdd_cdd(t2m_daily_data,country_mask)
    demand_timeseries = energy_model_functions.calc_national_wd_demand_2017(hdd,cdd,'inputs/ERA5_Regression_coeffs_demand_model.csv',country.replace(" ", "_"))
    demand_timeseries = energy_model_functions.convert_daily_demand_to_hourly(demand_timeseries,dates_daily)
    demand_timsseries = demand_timeseries/(-100.) # Convert to units of 100 GW and make negative for demand

    # Wind
    wind_data = energy_model_functions.load_10mwindspeed_and_convert_to_hub_height(ddir,'10mwd_1hrly_mean_a000_2018-01_2018-12.nc','item3249_1hrly_mean',hub_height)
    gridded_wind_power_class1 = energy_model_functions.convert_to_windpower(wind_data,'inputs/power_onshore.csv')
    country_wind_power_class1 = energy_model_functions.country_wind_power(gridded_wind_power_class1,'inputs/'+country.replace(" ","_")+'_windfarm_dist_hackathon.nc')

    return solar_pv_cf, hdd,cdd,demand_timeseries,country_wind_power_class1,dates,dates_daily

def main():
    # Set up the plot
    fig = plt.figure()
    fig.set_size_inches(10,8)
    
    i=0



    for country,code in countries.items():
        print(country,code)
        solar_pv_cf,hdd, cdd,demand, wind_power_cf,dates, dates_daily = calliope_inputs_country(country,hub_height)
        print(demand.shape)        
        if i==0:
            solar_pv_df=pd.DataFrame(data={'timestep':dates})
            demand_df=pd.DataFrame(data={'timestep':dates})
            wind_df=pd.DataFrame(data={'timestep':dates})
        solar_pv_df[code] = solar_pv_cf
        demand_df[code] = demand
        wind_df[code]=wind_power_cf
        i+=1

    ax1 = plt.subplot2grid((3,1),(0,0))
    ax2 = plt.subplot2grid((3,1),(1,0))
    ax3 = plt.subplot2grid((3,1),(2,0))

    ax1.set_title("Solar PV Capacity Factor")
    ax1.plot(solar_pv_df['GBR'],color='red',label="GBR")
    ax1.plot(solar_pv_df['IRL'],color='blue',label="IRL")
    ax1.set_xlim(0,8640)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(720))
    ax1.set_xticklabels(calendar.month_abbr)
    ll=plt.legend(loc="upper right",prop={"size": 14},fancybox=True,numpoints=1)

    ax2.set_title("Demand")
    ax2.plot(demand_df['GBR'], color='red',label="GBR")
    ax2.plot(demand_df['IRL'], color='blue',label="IRL")
    ax2.set_xlim(0,8640)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(720))
    ax2.set_xticklabels(calendar.month_abbr)
    ll=plt.legend(loc="upper right",prop={"size": 14},fancybox=True,numpoints=1)

    ax3.set_title("Wind Power Capacity Factor")
    ax3.plot(wind_df['GBR'], color='red',label="GBR")
    ax3.plot(wind_df['IRL'], color='blue',label="IRL")
    ax3.set_xlim(0,8640)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(720))
    ax3.set_xticklabels(calendar.month_abbr)
    ll=plt.legend(loc="upper right",prop={"size": 14},fancybox=True,numpoints=1)

    fig.canvas.draw()
    plt.tight_layout()
    fig.savefig(prefix+"_calliope_inputs.png")


    solar_pv_df.to_csv(prefix+'_solar_PV.csv',index=False)
    demand_df.to_csv(prefix+'_demand.csv',index=False)
    wind_df.to_csv(prefix+'_wind.csv',index=False)        

    print('Finished!')

#Washerboard function that allows main() to run on running this file
if __name__=="__main__":
      main()


