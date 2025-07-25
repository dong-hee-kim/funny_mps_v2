
from numba import jit
from functools import lru_cache
import sys
sys.path.insert(0, './script')
from curate_training_image import curate_training_image
import numpy as np
import time
import pyvista as pv
import pandas as pd

def visualize_3D_pyvista_with_well(realization, num_of_wells, well_locs):
    if len(well_locs) != num_of_wells:
        raise ValueError(f"well_locs: ({len(well_locs)}) are not mattch with number of wells: ({num_of_wells}).")
    grid = pv.ImageData()

    # Set grid dimensions (+1 because dimensions are points, not cells)
    grid.dimensions = np.array(realization.shape) + 1  # (nx+1, ny+1, nz+1)

    # Set spacing and origin
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    # Flatten the 3D facies values and assign to cell data
    grid.cell_data["Facies"] = realization.flatten(order="F")  # Fortran order for column-major

    # Create plotter and add volume
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="Facies", cmap="viridis", show_scalar_bar=True)
    
    # add wells
    z_top = realization.shape[2] + 5
    for i in range(num_of_wells):
        wx, wy = well_locs[i]
        start = (float(wx) + 0.5, float(wy) + 0.5, 0.0)
        end = (float(wx) + 0.5, float(wy) + 0.5, float(z_top))
        well_line = pv.Line(start, end)
        plotter.add_mesh(well_line, color="blue", line_width=5)
        plotter.add_point_labels([end], [f"well {i+1}"], font_size = 18, point_color = "red", text_color = "red")

    # Add axes and view settings
    plotter.show_axes()
    plotter.view_vector((1, 1, 1))
    plotter.add_text("X: Left-Right\nY: Front-Back\nZ: Bottom-Top", position="upper_left", font_size=10)
    plotter.show()


def generating_hard_data(TI_3D, well_loc_x, well_loc_y, real_nz, num_of_wells):

    unique_facies = list(np.unique(TI_3D).astype(np.int8))
    x_list = well_loc_x.get("x", [])
    y_list = well_loc_y.get("y", [])
    well_coords = list(zip(x_list, y_list))[:num_of_wells]

    hard_data_rows = []
    for (x, y) in well_coords:
        for z in range(real_nz // 2):  
            facies = np.random.choice(unique_facies)  
            hard_data_rows.append({'x': x, 'y': y, 'z': z, 'facies': facies})

    hard_data = pd.DataFrame(hard_data_rows)
    hard_data = hard_data.sort_values(by=['x', 'y', 'z'])
    
    return hard_data
    
@jit(nopython=True)
def fast_bincount(arr, minlength):
    """
    Numba-accelerated bincount for non-negative integer arrays.

    Args:
        arr (np.ndarray): 1D array of non-negative integers.
        minlength (int): The minimum number of bins.

    Returns:
        np.ndarray: The count for each bin. dtype=int64.

    Note:
        Equivalent to np.bincount(arr, minlength=minlength) but potentially faster
        when called repeatedly within Numba-compiled functions. Assumes arr
        contains values >= 0 and < minlength.
    """
    
    counts = np.zeros(minlength, dtype=np.int64)
    for val in arr:
        counts[val] += 1
    return counts

def predictive_model(data_x, data_y, 
                     input_x, 
                     facies_ratio, 
                     unique_facies) -> np.array:

    """
    Predicts facies probability based on matching patterns in training data.

    Finds rows in `data_x` that match the `input_x` (ignoring -1 values)
    and calculates the probability distribution of the corresponding `data_y`.
    Falls back to `facies_ratio` if no matches are found or input is all -1.

    Args:
        data_x (np.ndarray): The input features from the training image (data_x).
                            Shape (n_samples, n_features).
        data_y (np.ndarray): The target outputs from the training image (data_y).
                            Shape (n_samples, 1) or (n_samples,).
        input_x (np.ndarray): The pattern from the realization grid to predict.
                             Shape (1, n_features). Should contain -1 for unknown values.
        facies_ratio (np.ndarray): The overall proportion of each facies in the TI.
                                  Used as a fallback. Shape (n_facies,).
        unique_facies (np.ndarray): Array of unique facies codes present in the TI.
                                    Used to determine the size of the output distribution.

    Returns:
        np.ndarray: A probability distribution over the `unique_facies`. Shape (n_facies,).
    """
    
    if np.all(input_x == -1):
        return facies_ratio

    # Identify valid columns (features that are not -1)
    valid_cols = np.where(input_x != -1)[1]
    input_x_filtered = input_x[:, valid_cols]
    data_x_filtered = data_x[:, valid_cols]

    # Find matching rows
    matching_rows = np.all(data_x_filtered == input_x_filtered, axis=1)
    matched_labels = data_y[matching_rows]

    # Count occurrences using np.bincount
    counts = fast_bincount(matched_labels.astype(np.int32).flatten(), minlength=len(unique_facies))

    # If counts sum to zero, fall back to facies_ratio
    if counts.sum() == 0:
        return facies_ratio

    return counts / counts.sum()

def multi_points_modeling(TI_3D, 
                          template_size, 
                          random_seed, 
                          real_nx, real_ny, real_nz, 
                          hard_data,
                          soft_data = None,
                          verbose = False):

    """
    Generates a realization grid mimicking the patterns found in the Training Image (TI),
    optionally conditioned to hard data.

    Args:
        TI_3D (np.ndarray): The 3D Training Image (facies).
        template_size (list[int]): Dimensions (x, y, z) of the template. Must be odd.
        random_seed (int): Seed for the random number generator.
        real_nx (int): X-dimension of the desired realization grid.
        real_ny (int): Y-dimension of the desired realization grid.
        real_nz (int): Z-dimension of the desired realization grid.
        hard_data (np.ndarray, optional): Grid of known values to condition the simulation.
                                         Shape should match (real_nx, real_ny, real_nz).
                                         Use a sentinel value (e.g., -1) for unknown nodes.
                                         Defaults to None (unconditional simulation).
        soft_data (np.ndarray, optional): Not used. Defaults to None.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        np.ndarray: The generated realization grid with shape (real_nx, real_ny, real_nz).
    """
    realization, [facies_ratio, unique_facies], [data_x, data_y, flag], random_path = _preprocessing_MPS(TI_3D, 
                                                                                                    template_size,  
                                                                                                    real_nx, real_ny, real_nz, 
                                                                                                    hard_data = hard_data,
                                                                                                    verbose = verbose)
    padding_x, padding_y, padding_z = int((template_size[0]-1)/2), int((template_size[1]-1)/2), int((template_size[2]-1)/2)

    
    # for one iteration - randomly generate realization
    np.random.seed(random_seed)
    np.random.shuffle(random_path.T)
    if soft_data is None:
        return _run_mps(realization, facies_ratio, unique_facies, 
                        data_x, data_y, flag, random_path, 
                        padding_x, padding_y, padding_z, verbose = verbose)
    else:
        return _run_mps_w_soft_data(realization, facies_ratio, unique_facies, 
                        data_x, data_y, flag, random_path, 
                        padding_x, padding_y, padding_z, soft_data=soft_data,
                        verbose = verbose)
def _run_mps(realization, facies_ratio, unique_facies, 
             data_x, data_y, flag, random_path, 
             padding_x, padding_y, padding_z,
             verbose = False):
    """
    Run one iteration of the MPS simulation.

    Args:
        realization (np.ndarray): The realization grid to condition the simulation.
        facies_ratio (list): The proportion of each facies in the TI.
        unique_facies (list): The unique facies codes present in the TI.
        data_x (np.ndarray): The input features from the training image (data_x).
        data_y (np.ndarray): The target outputs from the training image (data_y).
        flag (np.ndarray): A boolean mask indicating which values in each template
                           are valid (not -1) and should be used for prediction.
        random_path (np.ndarray): A 3D array of random coordinates to visit in order.
        padding_x (int): X-padding of the template.
        padding_y (int): Y-padding of the template.
        padding_z (int): Z-padding of the template.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        np.ndarray: The updated realization grid with shape (real_nx, real_ny, real_nz)
    """
    if verbose:
        print("... starting [_run_mps]")
        start = time.time()

    def make_hashable_key(input_x_array):
        return input_x_array.tobytes()

    @lru_cache(maxsize=20000)
    def cached_predictive_model_from_bytes(input_x_bytes):
        input_x_array = np.frombuffer(input_x_bytes, dtype=np.int16).reshape(1, -1)
        return predictive_model(data_x, data_y, input_x_array, facies_ratio, unique_facies)

    for ii, jj, kk in zip(random_path[0].T, random_path[1].T, random_path[2].T):
        if realization[ii, jj, kk] != -1:
            continue

        # Efficient ravel and index slicing
        template = realization[
            ii - padding_x : ii + padding_x + 1,
            jj - padding_y : jj + padding_y + 1,
            kk - padding_z : kk + padding_z + 1
        ].ravel()

        input_x = template[flag]
        input_key = make_hashable_key(input_x.astype(np.int16))  # ensure dtype for consistent hashing
        prob = cached_predictive_model_from_bytes(input_key)

        realization[ii, jj, kk] = np.random.choice(unique_facies, p=prob)

    realization = _remove_padding(realization, padding_x, padding_y, padding_z)

    if verbose:
        end = time.time()
        print(f"==> finishing [_run_mps] in {end-start:.2f} seconds.")

    return realization

def _run_mps_w_soft_data(realization, facies_ratio, unique_facies, 
             data_x, data_y, flag, random_path, 
             padding_x, padding_y, padding_z, soft_data,
             verbose = False):
    """
    Run one iteration of the MPS simulation.

    Args:
        realization (np.ndarray): The realization grid to condition the simulation.
        facies_ratio (list): The proportion of each facies in the TI.
        unique_facies (list): The unique facies codes present in the TI.
        data_x (np.ndarray): The input features from the training image (data_x).
        data_y (np.ndarray): The target outputs from the training image (data_y).
        flag (np.ndarray): A boolean mask indicating which values in each template
                           are valid (not -1) and should be used for prediction.
        random_path (np.ndarray): A 3D array of random coordinates to visit in order.
        padding_x (int): X-padding of the template.
        padding_y (int): Y-padding of the template.
        padding_z (int): Z-padding of the template.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        np.ndarray: The updated realization grid with shape (real_nx, real_ny, real_nz)
    """
    if verbose:
        print("Running one iteration of the MPS simulation...")
        start = time.time()
    for ii, jj, kk in zip(random_path[0].T, random_path[1].T, random_path[2].T):
        if realization[ii, jj, kk] != -1:
            continue
        template = realization[ii-padding_x:ii+(padding_x+1),
                            jj-padding_y:jj+(padding_y+1),
                            kk-padding_z:kk+(padding_z+1)].copy().flatten()
        input_x = template[flag].reshape(1,-1)  
        tau = soft_data[ii-padding_x, jj-padding_y, kk-padding_z, :]

        prob = predictive_model(data_x, data_y,  input_x,  facies_ratio, unique_facies)*tau
        prob = prob/np.sum(prob)
        realization[ii, jj, kk] = np.random.choice(unique_facies, 
                                                p=prob)

    realization =  _remove_padding(realization, padding_x, padding_y, padding_z)
    if verbose:
        end = time.time()
        print(f"One iteration of the MPS simulation completed in {end-start:.2f} seconds.")
    return realization


def _remove_padding(realization, padding_x, padding_y, padding_z):
    """
    Removes padding from a 3D realization grid.

    Args:
        realization (np.ndarray): The padded realization grid.
        padding_x (int): Number of padding elements to remove from the x-dimension.
        padding_y (int): Number of padding elements to remove from the y-dimension.
        padding_z (int): Number of padding elements to remove from the z-dimension.

    Returns:
        np.ndarray: The realization grid with padding removed.
    """

    if padding_z != 0:
        return realization[padding_x:-padding_x, padding_y:-padding_y, padding_z:-padding_z]
    else:
        return realization[padding_x:-padding_x, padding_y:-padding_y]

def _preprocessing_MPS(TI_3D, 
                      template_size, 
                      real_nx, real_ny, real_nz, 
                      hard_data,
                      verbose = False):
    
    """
    Preprocess the Training Image (TI_3D) for MPS.

    Args:
        TI_3D (np.ndarray): The 3D Training Image (facies).
        template_size (list[int]): Dimensions (x, y, z) of the template. Must be odd.
        real_nx (int): X-dimension of the desired realization grid.
        real_ny (int): Y-dimension of the desired realization grid.
        real_nz (int): Z-dimension of the desired realization grid.
        hard_data (Optional[np.ndarray], optional): Grid of known values to condition the simulation.
                                                 Shape should match (real_nx, real_ny, real_nz).
                                                 Use a sentinel value (e.g., -1) for unknown nodes.
                                                 Defaults to None (unconditional simulation).
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        Tuple[np.ndarray, List, List, np.ndarray]: A tuple containing the padded realization grid,
                                                    a list of facies ratios and unique facies,
                                                    a list of the input features (data_x), target outputs (data_y), and a boolean mask (flag),
                                                    and a 3D array of random coordinates to visit in order.
    """
    unique_facies = list(np.unique(TI_3D).astype(np.int8))
    facies_ratio = [np.sum(TI_3D==f)/np.prod(TI_3D.shape) for f in unique_facies]
    padding_x, padding_y, padding_z = int((template_size[0]-1)/2), int((template_size[1]-1)/2), int((template_size[2]-1)/2)
    data_x, data_y, flag = curate_training_image(TI_3D, template_size, 1.0, verbose = verbose)
    # TODO: generate model
    realization = np.ones((real_nx+2*padding_x, real_ny+2*padding_y, real_nz+2*padding_z))*-1
    if hard_data is not None:
        if padding_z != 0:
            realization[padding_x:-padding_x, padding_y:-padding_y, padding_z:-padding_z] = hard_data
        else:
            realization[padding_x:-padding_x, padding_y:-padding_y] = hard_data
        if verbose:
            print('... [_preprocessing_MPS] hard data is conditioned')
    x_0, x_1 = int(0 +padding_x), int(realization.shape[0] - padding_x)
    y_0, y_1 = int(0 +padding_y), int(realization.shape[1] - padding_y)
    z_0, z_1 = int(0 +padding_z), int(realization.shape[2] - padding_z)
    xx, yy, zz = np.meshgrid(range(x_0, x_1), range(y_0, y_1), range(z_0, z_1))
    random_path = np.array([i.flatten() for i in [xx, yy, zz]])
    return realization, [facies_ratio, unique_facies], [data_x, data_y, flag], random_path

def multi_points_modeling_multi_scaled(TI, n_level, level_size,
                                      template_size, 
                                      random_seed, 
                                      real_nx, real_ny, real_nz, 
                                      hard_data, 
                                      verbose = False,
                                      return_multi_scale_real = False):
    
    TI_s, grid_size_s, real_s = [], [], []
    nx, ny, nz = real_nx, real_ny, real_nz

    for level in range(n_level):
        TI_s.append(TI[::level_size**level, ::level_size**level, ::round(level_size**level)])
        grid_size_s.append((nx, ny, nz))
        real_s.append(np.ones((nx, ny, nz)) * -1)  
        nx, ny, nz = round(nx / level_size), round(ny / level_size), round(nz / level_size)

    if hard_data is None:
        real = np.ones(grid_size_s[-1]) * -1
    else:
        real = np.ones(grid_size_s[-1]) * -1
        for _, row in hard_data.iterrows():
            x, y, z = int(row['x']), int(row['y']), int(row['z'])
            facies_val = row['facies']
            for level in range(n_level):
                nx, ny, nz = grid_size_s[level]
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                    real_s[level][x, y, z] = facies_val
                x, y, z = round(x / level_size), round(y / level_size), round(z / level_size)
        real = real_s[-1]

    if verbose:
        print('[MPS] multi-scale MPS starts')

    for idx, (level, TI_at_level, grid_size_at_level) in enumerate(zip(range(n_level)[::-1],TI_s[::-1], grid_size_s[::-1])):
        if verbose:
            print("---"*10)
            print(f'<Scale {level} start> Grid size is {grid_size_at_level}')

        real = multi_points_modeling(TI_at_level, 
                                    template_size, 
                                    random_seed, 
                                    grid_size_at_level[0], grid_size_at_level[1], grid_size_at_level[2], 
                                    real, 
                                    verbose=verbose)
        real_s.append(real)
        
        if verbose:
            print(f'<Scale {level} start> Done')
        if level == 1:
            break 
        real_next = np.ones(grid_size_s[level-1]) * -1
        real_next[1::level_size, 1::level_size, 1::level_size] = real

        if hard_data is not None:
            for _, row in hard_data.iterrows():
                x, y, z = int(row['x']), int(row['y']), int(row['z'])
                facies_val = row['facies']
                if 0 <= x < real_next.shape[0] and 0 <= y < real_next.shape[1] and 0 <= z < real_next.shape[2]:
                    if real_next[x, y, z] == -1:
                        real_next[x, y, z] = facies_val

        real = real_next.copy()
    
    if return_multi_scale_real:
        return real_s
    else:
        return real