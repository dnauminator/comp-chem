# this script parses amber mdout files to extract the total energy
import argparse
import os
import glob

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This script parses amber mdout files to extract the total energy.")
    parser.add_argument("path", help="The filepath for the file to be analyzed."
                        "Use '*.mdout' to process all .mdout files in the 'data' folder.", nargs ='*')
    parser.add_argument('-img_plot', help='If specified, will generate an image plot of the total energy over time.', action='store_true')
    data_folder = 'data'

    args = parser.parse_args()
    
    if args.path == ['*.mdout']:
        path = glob.glob(os.path.join(data_folder, '*.mdout'))
    else:
        path = args.path


    for current_file_name in path:
        if os.path.isabs(current_file_name) or current_file_name.startswith(data_folder):
            prod = current_file_name
        else:
            prod = os.path.join(data_folder, current_file_name)
        base_output = os.path.basename(current_file_name)
        # Remove the .mdout extension if present for cleaner output names
        if base_output.lower().endswith('.mdout'):
            base_output = base_output[:-len('.mdout')]

        output_filename = f'{base_output}_Etot.txt'
        datafile = open(output_filename, 'w+')
        outfile = open(prod,'r')
        data = outfile.readlines()
        outfile.close()
        linewriter = []
        for line in data:
            if 'Etot   ' in line:
                Etot_line = line
                words = Etot_line.split()
                Etot = float(words[2])
                linewriter.append(F'{Etot} \n')
        linewriter = linewriter[:-2] # prune the last two lines (extraneous data)
        if args.img_plot:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create a time array based on the number of energy entries
            time = np.arange(len(linewriter))

            # Convert linewriter to a numpy array for plotting
            energies = np.array([float(energy.strip()) for energy in linewriter])

            # Plotting the total energy over time
            plt.figure(figsize=(10, 5))
            plt.plot(time, energies, marker='o', linestyle='-', color='b')
            plt.title(f'Total Energy Over Time ({base_output})')
            plt.xlabel('Time Step')
            plt.ylabel('Total Energy (Etot)')
            plt.grid()
            plt.savefig(f'{base_output}_Etot_plot.png')
            plt.close()
        for line_to_write in linewriter:
            datafile.write(F'{line_to_write}')
        datafile.close()