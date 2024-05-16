import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage

from TumorGrowthToolkit.FK_2c import Solver


# Plotting function
def plot_tumor_states(wm_data, initial_states, final_states, slice_index, cmap1, cmap2, cmap3):
    plt.figure(figsize=(18, 6))  # Adjusted figure size for 3 columns

    # Fields to plot
    fields = ['P', 'N', 'S']
    titles = ['Proliferative Cells', 'Necrotic Cells', 'Nutrient Field']

    # Plotting initial states
    for i, field in enumerate(fields):
        plt.subplot(2, 3, i + 1)
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(initial_states[field][:, :, slice_index], cmap=cmap2 if i == 0 else cmap3, vmin=0, vmax=1,
                   alpha=0.65)
        plt.title(f"Initial {titles[i]}")

    # Plotting final states
    for i, field in enumerate(fields):
        plt.subplot(2, 3, i + 4)
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(final_states[field][:, :, slice_index], cmap=cmap2 if i == 0 else cmap3, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Final {titles[i]}")

    plt.tight_layout()
    plt.show()


def plot_time_series(gm_data, wm_data, time_series_data, slice_index, cmap1, cmap2):
    plt.figure(figsize=(8, 24))

    # Fields to plot
    fields = ['P', 'N', 'S']
    field_titles = ['Proliferative', 'Necrotic', 'Nutrient']

    # Generate indices for selected timesteps
    num_timesteps = np.array(time_series_data['P']).shape[0]
    time_points = np.linspace(0, num_timesteps - 1, 8, dtype=int)
    time_max = num_timesteps - 1
    th_plot = 0.1
    margin_x = 25
    margin_y = 25
    for i, t in enumerate(time_points):
        # Calculate the relative time (0 to 1)
        relative_time = t / time_max

        for j, field in enumerate(fields):
            ax = plt.subplot(len(time_points), 3, i * 3 + j + 1)

            # Plot the white matter data
            plt.contourf(
                np.fliplr(np.flipud(np.rot90(gm_data[margin_x:-margin_x, margin_y:-margin_y, slice_index], -1))),
                levels=[0.5, 1], colors='gray', alpha=0.35)
            # Plot the field data
            vol = np.array(time_series_data[field])[t, margin_x:-margin_x, margin_y:-margin_y, slice_index]
            vol_display = np.array(np.where(vol > th_plot, vol, np.nan))
            plt.imshow(np.fliplr(np.flipud(np.rot90(vol_display, -1))), cmap=cmap2, vmin=0, vmax=1, alpha=1.)

            # Add field titles
            plt.title(f"{field_titles[j]}")

            # Add time annotation only once per row, above the subplot
            if j == 0:
                ax.text(0.5, 1.20, f"Time: {relative_time:.2f}", transform=ax.transAxes, fontsize=12, fontweight='bold',
                        ha='center', va='center')

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


# Create binary segmentation masks
wm_data = nib.load('mehdi_data/wm.nii.gz').get_fdata()
gm_data = nib.load('mehdi_data/gm.nii.gz').get_fdata()


# Set up parameters
parameters = {
    'Dw': 0.9,          # Diffusion coefficient for the white matter
    'rho': 0.14,         # Proliferation rate
    'lambda_np': 0.35, # Transition rate between proli and necrotic cells
    'sigma_np': 0.5, #Transition threshols between proli and necrotic given nutrient field
    'D_s': 1.3,      # Diffusion coefficient for the nutrient field
    'lambda_s': 0.05, # Proli cells nutrients consumption rate
    'RatioDw_Dg': 100,  # Ratio of diffusion coefficients in white and grey matter
    'Nt_multiplier': 8,
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': 0.35,    # tumor position [%]
    'NyT1_pct': 0.6,
    'NzT1_pct': 0.5,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': .5, #resultion scaling for calculations
    'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
    'verbose': True, #printing timesteps
    'time_series_solution_Nt': 8 # number of timesteps in the output
}

# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = Solver(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference
print(f"Execution Time: {execution_time} seconds")

# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])
# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

if result['success']:
    print("Simulation successful!")
    # Extract initial and final states from the result
    initial_states = result['initial_state']
    final_states = result['final_state']
    plot_tumor_states(wm_data, initial_states, final_states, NzT, cmap1, cmap2, cmap2)
    time_series_data = result['time_series']
    plot_time_series(gm_data, wm_data, time_series_data, NzT, cmap1, cmap2)
else:
    print("Error occurred:", result['error'])

#### Saving outputs
write_name_n = './mehdi_data/simulation_2c_N.nii.gz'
write_name_p = './mehdi_data/simulation_2c_P.nii.gz'
write_name_s = './mehdi_data/simulation_2c_S.nii.gz'
src_nib = nib.load('./mehdi_data/t1.nii.gz')
dst_nib_n = nib.Nifti1Image(result['final_state']['N'], src_nib.affine, src_nib.header)
dst_nib_p = nib.Nifti1Image(result['final_state']['P'], src_nib.affine, src_nib.header)
dst_nib_s = nib.Nifti1Image(result['final_state']['S'], src_nib.affine, src_nib.header)
nib.save(dst_nib_n, write_name_n)
nib.save(dst_nib_p, write_name_p)
nib.save(dst_nib_s, write_name_s)
