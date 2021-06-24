#############################################################################
# Program : calc_calliope_inouts.py
# Author  : Hannah Bloomfield
# Purpose : Calculate energy variables from climate data
# Modified: 22/07/2021 Sarah Sparrow to ajust for CPDN interface
#############################################################################

import numpy as np
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
from netCDF4 import num2date
import datetime
import shapely.geometry

# Common functions --------------------------------------------------------------------

def load_country_weather_data(COUNTRY,data_dir,filename,nc_key,nc_time,convert_to_daily):

    """
    This function takes the ERA5 reanalysis data, loads it and applied a 
    country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints 
    set to zeros.

    You will need the shpreader.natural_earth data downloaded 
    to find the shapefiles.

    Args:
        COUNTRY (str): This must be a name of a country (or set of) e.g. 
            'United Kingdom','France','Czech Republic'

        data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The string you need to load the .nc data 
            e.g. 't2m','rsds'

        convert_to_daily (int): For no conversion enter 0 otherwise enter
            the number of data points in 1 day.  e.g. 24 for hourly, 4 for 6 hourly. 

    Returns:

        country_masked_data (array): Country-masked weather data, dimensions 
            [time,lat,lon] where there are 0's in locations where the data is 
            not within the country border.

        MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if 
           the data is within a country border and zeros if data is outside a 
           country border. 

    """


    # first loop through the countries and extract the appropraite shapefile
    countries_shp = shpreader.natural_earth(resolution='10m',category='cultural',
                                            name='admin_0_countries')
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes['NAME_LONG'] == COUNTRY:
            print('Found country: '+COUNTRY)
            country_shapely.append(country.geometry)
    
    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str,mode='r')
    lons = dataset.variables['longitude'][:]
    lats = dataset.variables['latitude'][:]
    time = dataset.variables[nc_time]
    sdate=num2date(time[0],time.units,time.calendar)
    start_date=datetime.datetime.strptime(str(sdate.year)+"/"+str(sdate.month)+"/"+str(sdate.day)+" 00:00:00", "%Y/%m/%d %H:%M:%S")
    times = [start_date + datetime.timedelta(hours=i) for i in range(len(time[:]))]
    data = dataset.variables[nc_key][:] # data in shape [time,lat,lon]
    dataset.close()
    
    # get data in appropriate units for models
    if nc_key in ['t2m','item3236_1hrly_mean']:
        data = data-273.15 # convert to Kelvin from Celsius
    if nc_key == 'ssrd':
        data = data/3600. # convert Jh-1m-2 to Wm-2
    if nc_key =='field203':
        # As CPDN rsds output is 1 every 3 hours rather than hourly we need to repeat the rsds values 
        data=np.repeat(data,3,axis=0)
        print(data.shape)

    if convert_to_daily > 0: # if hourly data convert to daily
        data = np.mean ( np.reshape(data, (int(len(data)/convert_to_daily), int(convert_to_daily),len(lats),len(lons))),axis=1)
        times = np.array([start_date + datetime.timedelta(days=i) for i in range(int(len(data)))])
        print('Converting to daily-mean')

    LONS, LATS = np.meshgrid(lons,lats) # make grids of the lat and lon data
    x, y = LONS.flatten(), LATS.flatten() # flatten these to make it easier to 
    #loop over.
    points = np.vstack((x,y)).T
    MASK_MATRIX = np.zeros((len(x),1))
    # loop through all the lat/lon combinations to get the masked points
    for i in range(0,len(x)):
        my_point = shapely.geometry.Point(x[i],y[i]) 
        if country_shapely[0].contains(my_point) == True: 
            MASK_MATRIX[i,0] = 1.0 # creates 1s and 0s where the country is
    
    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX,(len(lats),len(lons)))

    # now apply the mask to the data that has been loaded in:

    country_masked_data = data*MASK_MATRIX_RESHAPE
                                     


    return(country_masked_data,MASK_MATRIX_RESHAPE,lons,lats,times)

# Solar functions ----------------------------------------------------------------

def solar_PV_model(country_masked_data_T2m,country_masked_data_ssrd,country_mask):

    """

    This function takes in arrays of country_masked 2m temperature (celsius)
    and surface solar irradiance (Wm-2) and converts this into a time series
    of solar power capacity factor using the method from Bloomfield et al.,
    (2020) https://doi.org/10.1002/met.1858

    Args:

        country_masked_data_T2m (array): array of 2m temperatures, Dimensions 
            [time, lat,lon] or [lat,lon] in units of celsius.
        country_masked_data_ssrd (array): array of surface solar irradiance, 
            Dimensions [time, lat,lon] or [lat,lon]in units of Wm-2.
        country_mask (array): dimensions [lat,lon] with 1's within a country 
            border and 0 outside of it. 
    Returns:

        spatial_mean_solar_cf (array): Dimesions [time], Timeseries of solar 
            power capacity factor, varying between 0 and 1.


    """
   # reference values, see Evans and Florschuetz, (1977)
    T_ref = 25. 
    eff_ref = 0.9 #adapted based on Bett and Thornton (2016)
    beta_ref = 0.0042
    G_ref = 1000.
 
    rel_efficiency_of_pannel = eff_ref*(1 - beta_ref*(country_masked_data_T2m - T_ref))
    capacity_factor_of_pannel = np.nan_to_num(rel_efficiency_of_pannel*
                                              (country_masked_data_ssrd/G_ref)) 


    spatial_mean_solar_cf = np.zeros([len(capacity_factor_of_pannel)])
    for i in range(0,len(capacity_factor_of_pannel)):
        capacity_factor_of_pannel=capacity_factor_of_pannel.squeeze()
        spatial_mean_solar_cf[i] = np.average(capacity_factor_of_pannel[i,:,:],
                                        weights=country_mask)

    return(spatial_mean_solar_cf)

# Demand functions ---------------------------------------------------------------

def calc_hdd_cdd(t2m_array,country_mask):

    """

    This function takes in an array of country_masked 2m temperature (celsius)
    and converts this into a time series of heating-degree days (HDD) and cooling
    degree days (CDD) using the method from Bloomfield et al.,(2020) 
    https://doi.org/10.1002/met.1858

    Args:

        t2m_array (array): array of country_masked 2m temperatures, Dimensions 
            [time, lat,lon] or [lat,lon] in units of celsius. 
        country_mask (array): array of the country mask applied to the t2m data 
            Dimensions [lat,lon] with 1's for gridpoints within the country.
    Returns:

        HDD_term (array): Dimesions [time], timeseries of heating degree days
        CDD_term (array): Dimesions [time], timeseries of cooling degree days


    """
    len_time = np.shape(t2m_array)[0]

    spatial_mean_t2m =np.zeros(len_time)
    for i in range(0,len_time):
        spatial_mean_t2m[i] = np.average(t2m_array[i,:,:],weights=country_mask)

    # note the function works on daily temperatures. so make sure these are daily!

    HDD_term = np.zeros(len_time)
    CDD_term = np.zeros(len_time)

    for i in range(0,len_time):
        if spatial_mean_t2m[i] <= 15.5:
            HDD_term[i] = 15.5 - spatial_mean_t2m[i]
        else:
            HDD_term[i] =0

    for i in range(0,len_time):
        if spatial_mean_t2m[i] >= 22.0:
            CDD_term[i] = spatial_mean_t2m[i] - 22.0
        else:
            CDD_term[i] =0


    return(HDD_term,CDD_term)


def calc_national_wd_demand_2017(hdd,cdd,filestr_reg_coefficients,COUNTRY):


    """

    This function takes in arrays of national heating-degree days (HDD) 
    and cooling degree days (CDD) using the method from Bloomfield et al.,(2020) 
    https://doi.org/10.1002/met.1858 Combines these with the published 
    regression coefficients to produce weather-dependent demand.
    Regression coefficients are available here for the ERA5 hourly demand model
    https://researchdata.reading.ac.uk/272/

    Args:

        hdd (array): array of national heating degree days, Dimensions 
            [time] 
        cdd (array): array of national cooling degree days, Dimensions 
            [time] 
        filestr_reg_coefficients (string): the filepath of the regression
            coeffients for the dmeand model published here: 
            http://dx.doi.org/10.17864/1947.272
        COUNTRY (string): The country name you wish to calculate demand for
            note that spaces should be underscores e.g. 'Czech_Republic'
           Only the 28 countries that have been modelled in the paper above
           are available.


    """

    all_coeffs = np.genfromtxt(filestr_reg_coefficients,skip_header=1,
                               delimiter=',')
    time_point = 2017. # this is the year the demand model is setup to 
                       # recreate data from.

    # Dictionary saying which country is in which column of the regression
    # coefficent file, filestr_reg_coefficients.
    column_dictionary = {
        "Austria" : 1,
        "Belgium" : 2,
        "Bulgaria" : 3,
        "Croatia" : 4,
        "Czech_Republic" : 5,
        "Denmark" : 6,
        "Finland" : 7,
        "France" : 8,
        "Germany" : 9,
        "Greece" : 10,
        "Hungary" : 11,
        "Ireland" : 12,
        "Italy" : 13,
        "Latvia" : 14,
        "Lithuania" : 15,
        "Luxembourg" : 16,
        "Montenegro" : 17,
        "Netherlands" : 18,
        "Norway" : 19,
        "Poland" : 20,
        "Portugal" : 21,
        "Romania" : 22,
        "Slovakia" : 23,
        "Slovenia" : 24,
        "Spain" : 25,
        "Sweden" : 26,
        "Switzerland" : 27,
        "United_Kingdom" : 28,

    }

    column = column_dictionary[COUNTRY]
    reg_coeffs = all_coeffs[:,column]

    time_coeff = reg_coeffs[0]
    hdd_coeff = reg_coeffs[8]
    cdd_coeff = reg_coeffs[9]
    #weekday_coeff = reg_coeffs[1]

    demand_timeseries = (time_coeff*time_point) + (hdd_coeff*hdd) + (cdd_coeff*cdd)
                  



    return(demand_timeseries)


# Wind functions ----------------------------------------------------------

def load_10mwindspeed_and_convert_to_hub_height(data_dir,filename,nc_key,hub_height):

    """
    This function takes the ERA5 reanalysis data, loads it and applied a 
    country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints 
    set to zeros.

    You will need the shpreader.natural_earth data downloaded 
    to find the shapefiles.

    Args:

        data_dir (str): The path for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The name of the 10m wind speed variable in the netcdf file
        
        hub_height (int): The hub height for the  wind turbine in m e.g. 100 
    Returns:

        wind_speed_data (array): wind speed data at hub height, dimensions 
            [time,lat,lon].

    """

  
    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str,mode='r')
    lons = dataset.variables['longitude'][:]
    lats = dataset.variables['latitude'][:]
    data = dataset.variables[nc_key][:] # data in shape [time,lat,lon]
    dataset.close()
    
    scale_factor=(hub_height/10.)**(1./7.)
    print(scale_factor)
    wind_speed_data = data*scale_factor

    return(wind_speed_data)



def convert_to_windpower(wind_speed_data,power_curve_file):

    """
    This function takes the ERA5 reanalysis data, loads it and applied a 
    country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints 
    set to zeros.

    You will need the shpreader.natural_earth data downloaded 
    to find the shapefiles.

    Args:

        gridded_wind_power (array): wind power capacity factor data, dimensions 
            [time,lat,lon]. Capacity factors range between 0 and 1.

        power_curve_file (str): The filename of a .csv file
            containing the wind speeds (column 0) and capacity factors 
            (column 2) of the chosen wind turbine.

    Returns:

        wind_power_cf (array): Gridded wind Power capacity factor  
            data, dimensions [time,lat,lon]. Values vary between 0 and 1.

    """

    # first load in the power curve data
    pc_w = []
    pc_p = []

    with open(power_curve_file) as f:
        for line in f:
            columns = line.split()
            #print columns[0]
            pc_p.append(np.float(columns[1]))  
            pc_w.append(np.float(columns[0]))  # get power curve output (CF)

    # convert to an array
    power_curve_w = np.array(pc_w)
    power_curve_p = np.array(pc_p)

    #interpolate to fine resolution.
    pc_winds = np.linspace(0,50,501) # make it finer resolution 
    pc_power = np.interp(pc_winds,power_curve_w,power_curve_p)

    reshaped_speed = wind_speed_data.flatten()
    test = np.digitize(reshaped_speed,pc_winds,right=False) # indexing starts 
    #from 1 so needs -1: 0 in the next bit to start from the lowest bin.
    test[test ==len(pc_winds)] = 500 # make sure the bins don't go off the 
    #end (power is zero by then anyway)
    wind_power_flattened = 0.5*(pc_power[test-1]+pc_power[test])

    wind_power_cf = np.reshape(wind_power_flattened,(np.shape(wind_speed_data)))
    
    return(wind_power_cf)

def country_wind_power(gridded_wind_power,wind_turbine_locations):

    """
    This function takes the ERA5 reanalysis data, loads it and applied a 
    country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints 
    set to zeros.

    You will need the shpreader.natural_earth data downloaded 
    to find the shapefiles.

    Args:

        gridded_wind_power (array): wind power capacity factor data, dimensions 
            [time,lat,lon]. Capacity factors range between 0 and 1.

        wind turbine locations (str): The filename of a .nc file
            containing the amount of installed wind power capacity in gridbox

    Returns:

        wind_power_country_cf (array): Time series of wind Power capacity factor
                      data, weighted by the installed capacity in each reanalysis
            gridbox from thewindpower.net database. dimensions [time]. 
            Values vary between 0 and 1.

    """

    # first load in the installed capacity data.

 
    dataset_1 = Dataset(wind_turbine_locations,mode='r')
    total_MW = dataset_1.variables['totals'][:]
    dataset_1.close()

    len_timeseries = np.shape(gridded_wind_power)[0]

    wind_power_country_cf = np.zeros(len_timeseries)

    for i in range(0,len_timeseries):
        wind_power_weighted = gridded_wind_power[i,:,:]*total_MW
        wind_power_country_cf[i] = np.sum(wind_power_weighted)/np.sum(total_MW)

    
    return(wind_power_country_cf)



def calc_diurnal_cycle(file1_str,file2_str,data_frequency):
    """ load in hourly demand data and calculate a diurnal cycle in demand"""
    actual_demand=[]

    #os.chdir(data_directory)
    with open(file1_str) as f:
        next(f) # gets around the problem with the header of the file

        counter = 0

        for line in f:
            counter = counter +1
            columns = line.split(',')
            if columns[2] == '"N/A"':
                actual_demand.append(actual_demand[counter-2])
            elif columns[2] == '"N/A"\n':
                actual_demand.append(actual_demand[counter-2])
            else:
                actual_demand.append(float(columns[2][1:-2])/1000) # append the actual load remember zero indexing.

    f.close()

    with open(file2_str) as f:
        next(f) # gets around the problem with the header of the file

        counter = 0
        for line in f:
            counter = counter +1
            columns = line.split(',')
            if columns[2] == '"N/A"':
                actual_demand.append(actual_demand[counter-2]) #replace with previous value so we dont have problems with Nans in regression model
            elif columns[2] == '"N/A"\n':
                actual_demand.append(actual_demand[counter-2])

            else:
                actual_demand.append(float(columns[2][1:-2])/1000) # append the actual load (this is half hourly data) remember zero indexing.

    f.close()

    # convert to hourly data
    if data_frequency == '15min':
        divider_for_demand = 4
    elif data_frequency == '30min':
        divider_for_demand = 2
    elif data_frequency == 'hourly':
        divider_for_demand = 1
    else:
        print("we dont have hourly demand...")

    len_of_training_period = int(len(actual_demand)/divider_for_demand) # then we have the length of everything. Note casting to int as array dimensions cannot be floats
    #print(len(actual_demand),len_of_training_period,divider_for_demand)

    actual_demand_array = np.array(actual_demand)# lets turn this list into an array.
    actual_demand_reshaped_array = np.reshape(actual_demand_array[0:len_of_training_period*divider_for_demand],[len_of_training_period,divider_for_demand]) # and reshape it so we can get daily mean demand
    actual_hourly_demand = np.mean(actual_demand_reshaped_array,axis=1)

    # now lets reshape to get the two years beside eachother (ignore leap day)
    both_actual_hourly_demand = np.reshape(actual_hourly_demand[0:2*8760],[2,8760])

    # lets roll the data so that winters are together
    rolled_actual_demand = np.roll(both_actual_hourly_demand,31*24,axis=1) # the year now starts at december.

    # now we need to reshape this again into a hourly timeseries
    reshaped_for_anomaly_calc = np.reshape(rolled_actual_demand,[2,365,24])

    # calculate the total mean demand
    mean_demand = np.mean(reshaped_for_anomaly_calc)

    # calculate diurnal cycle
    winter_anomaly = np.nanmean(np.nanmean(reshaped_for_anomaly_calc[:,0:91,:],axis=0),axis=0) - np.nanmean(reshaped_for_anomaly_calc[:,0:91,:])
    spring_anomaly = np.nanmean(np.nanmean(reshaped_for_anomaly_calc[:,91:182,:],axis=0),axis=0) - np.nanmean(reshaped_for_anomaly_calc[:,91:182,:])
    summer_anomaly = np.nanmean(np.nanmean(reshaped_for_anomaly_calc[:,182:243,:],axis=0),axis=0) -np.nanmean(reshaped_for_anomaly_calc[:,182:243,:])
    autumn_anomaly = np.nanmean(np.nanmean(reshaped_for_anomaly_calc[:,243:365,:],axis=0),axis=0) - np.nanmean(reshaped_for_anomaly_calc[:,243:365,:])

    # make a blended seasonal cycle of the form diurnal_cycle = diurnal_cycle(month,hour of the day)
    diurnal_cycle = np.zeros([12,24])
    diurnal_cycle[0,:] = winter_anomaly # January
    diurnal_cycle[1,:] = 0.5*winter_anomaly + 0.5*spring_anomaly
    diurnal_cycle[2,:] = 0.5*winter_anomaly + 0.5*spring_anomaly
    diurnal_cycle[3,:] = spring_anomaly #April
    diurnal_cycle[4,:] = 0.5*spring_anomaly + 0.5*summer_anomaly
    diurnal_cycle[5,:] = 0.5*spring_anomaly + 0.5*summer_anomaly
    diurnal_cycle[6,:] = summer_anomaly # July
    diurnal_cycle[7,:] = 0.5*summer_anomaly + 0.5*autumn_anomaly
    diurnal_cycle[8,:] = 0.5*summer_anomaly + 0.5*autumn_anomaly
    diurnal_cycle[9,:] = autumn_anomaly
    diurnal_cycle[10,:] = 0.5*autumn_anomaly + 0.5*winter_anomaly
    diurnal_cycle[11,:] = 0.5*autumn_anomaly + 0.5*winter_anomaly

    return diurnal_cycle, mean_demand


def convert_daily_demand_to_hourly(demand_timeseries,dates):
    """ Convert daily demand data to hourly by adding a diurnal cycle. The form of the diurnal cycle also varies with the calendar
    month"""

    # these files only apply to the UK, we will apply a fudge factor, scaling by the mean to apply this to Ireland or other countries
    file1_str = '/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_7/energy_conversion/group_7/inputs/Total_Load_Day_Ahead_Actual_201601010000-201701010000.csv'
    file2_str = '/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_7/energy_conversion/group_7/inputs/Total_Load_Day_Ahead_Actual_201701010000-201801010000.csv'
    diurnal_cycle, mean_demand_UK = calc_diurnal_cycle(file1_str,file2_str,data_frequency='30min')

    n_times = demand_timeseries.shape[0]
    mean_demand_country = np.mean(demand_timeseries)
    demand_timeseries_hourly = np.zeros([n_times,24])
    scaling = mean_demand_country / mean_demand_UK # fudge factor to scale the diurnal cycle by the mean demand

    for i,value in enumerate(demand_timeseries):
        month_index = dates[i].timetuple()[1] - 1 # -1 for zero indexing, hence Jan=0,Feb=1...
        demand_timeseries_hourly[i,:] = np.array([demand_timeseries[i]]).reshape(1,1) + scaling * diurnal_cycle[month_index,:].reshape(1,24) 

    demand_timeseries_hourly = demand_timeseries_hourly.flatten()

    return demand_timeseries_hourly

