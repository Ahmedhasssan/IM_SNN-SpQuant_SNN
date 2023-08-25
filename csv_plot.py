import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def make_csv_plot(png_dir,csv_files):
    files = os.listdir(csv_files)
    file = [file for file in files if file.endswith('.csv')]
    for i,fi in enumerate(file):
        data = 0
        data = pd.read_csv(csv_files+fi)
        plt.clf()
        #plt.xticks(np.arange(0, 400, 50))
        #fig, (ax1, ax2) = plt.subplots(2, 1) 
        plt.figure(figsize=(18, 6))
        #plt.subplot(2,1,1)
        plt.plot(data['t_avg'], marker='o', markersize=4, markeredgecolor='black', markerfacecolor='red')
        #plt.legend("P")
        plt.grid(b=True, which='both', axis='both',
            color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Channels')
        plt.ylabel('Spiking Rate')
        #plt.ylabel('Pixel distribution')
        # plt.subplot(2,1,2)
        # plt.plot(data['neg_pixels'], marker='x', markersize=4, markeredgecolor='black', markerfacecolor='red')
        # plt.legend("N")
        # plt.grid(b=True, which='both', axis='both',
        #     color='black', linestyle='-', linewidth=0.5)
        # plt.xlabel('Channels')
        # #plt.ylabel('Spiking Rate')
        # plt.ylabel('Pixel distribution')
        plt.title('Output Spiking activity')
        # plt.xlim([0,200])
        # plt.ylim([0,100])
        plt.show()
        # Save the plot.
        plt.savefig(png_dir+'Spiking Activity'+str(i)+'.png')
if __name__ == "__main__":
  csv_files = "/home/ahasssan/QESNN/new_pruning/QESNN/spike_distribution_prunned/"
  png_dir = "/home/ahasssan/QESNN/new_pruning/QESNN/png_files/spike_output_data_prunned/"
  # Convert the .log file to a .csv file.
  make_csv_plot(png_dir, csv_files)
  print("Conversion complete!")