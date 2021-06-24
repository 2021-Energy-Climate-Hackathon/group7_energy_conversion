#############################################################################
# Program : plot_ERA5_CPDN_comparison.py
# Author  : Sarah Sparrow
# Date    : 23/06/2021
# Purpose : Plot comparison of ERA5 and CPDN derived energy variables
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_dist(CPDN_data,ERA5_data,country,title,ivar,iplt):
    # Plot comparison distributions
    ax = plt.subplot2grid((3,2),(ivar,iplt))
    plt.setp(ax.get_xticklabels(),fontsize=16)
    plt.setp(ax.get_yticklabels(),fontsize=16)

    sns.distplot(CPDN_data[country], kde=True, color="Gold",label="CPDN",kde_kws={"linewidth":2.5,"color":"Gold"})
    sns.distplot(ERA5_data[country], kde=True, color="royalblue",label="ERA5",kde_kws={"linewidth":2.5,"color":"royalblue"})

    ax.set_title(country+" "+title)
    ax.set_xlabel(title)
    ax.set_ylabel("Density")

    ll=ax.legend(loc="upper right",prop={"size": 10},fancybox=True,numpoints=1)

def main():
    # Set up the plot
    fig = plt.figure()
    fig.set_size_inches(9,9)

    # Read in the energy variables to pandas data frame 
    CPDN_solar=pd.read_csv("CPDN_solar_PV.csv")
    ERA5_solar=pd.read_csv("ERA5_solar_PV.csv")
    
    CPDN_demand=pd.read_csv("CPDN_demand.csv")
    ERA5_demand=pd.read_csv("ERA5_demand.csv")

    CPDN_wind=pd.read_csv("CPDN_wind.csv")
    ERA5_wind=pd.read_csv("ERA5_wind.csv")

    plot_dist(CPDN_solar,ERA5_solar,"GBR","Solar Capacity Factor",0,0)
    plot_dist(CPDN_solar,ERA5_solar,"IRE","Solar Capacity Factor",0,1)

    plot_dist(CPDN_demand,ERA5_demand,"GBR","Demand",1,0)
    plot_dist(CPDN_demand,ERA5_demand,"IRE","Demand",1,1)

    plot_dist(CPDN_wind,ERA5_wind,"GBR","Wind Capacity Factor",2,0)
    plot_dist(CPDN_wind,ERA5_wind,"IRE","Wind Capacity Factor",2,1)


    plt.tight_layout()
    fig.savefig("ERA5_CPDN_comparison.png")

    print('Finished!')



#Washerboard function that allows main() to run on running this file
if __name__=="__main__":
      main()


