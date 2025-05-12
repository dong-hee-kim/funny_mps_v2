
from numba import jit
import numpy as np
import sys
sys.path.insert(0, './script')
from curate_training_image import curate_training_image
import time

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
                          hard_data = None,
                          soft_data = None,
                          verbose = False):

    """
    Generates a realization grid mimicking the patterns found in the Training Image (TI),
    optionally conditioned to hard data.

    Args:
        TI (np.ndarray): The 3D Training Image (facies).
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
        print("Running one iteration of the MPS simulation...")
        start = time.time()
    for ii, jj, kk in zip(random_path[0].T, random_path[1].T, random_path[2].T):
        if realization[ii, jj, kk] != -1:
            continue
        template = realization[ii-padding_x:ii+(padding_x+1),
                            jj-padding_y:jj+(padding_y+1),
                            kk-padding_z:kk+(padding_z+1)].copy().flatten()
        input_x = template[flag].reshape(1,-1)  
        realization[ii, jj, kk] = np.random.choice(unique_facies, 
                                                   p=predictive_model(data_x, 
                                                                      data_y, 
                                                                      input_x, 
                                                                      facies_ratio, 
                                                                      unique_facies))
    realization =  _remove_padding(realization, padding_x, padding_y, padding_z)
    if verbose:
        end = time.time()
        print(f"One iteration of the MPS simulation completed in {end-start:.2f} seconds.")
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

        prob = predictive_model(data_x, data_y, input_x, facies_ratio, unique_facies)*tau
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
                      hard_data = None,
                      verbose = False):
    
    """
    Preprocess the Training Image (TI) for MPS.

    Args:
        TI (np.ndarray): The 3D Training Image (facies).
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
    data_x, data_y, flag = curate_training_image(TI_3D, template_size, 1.0)

    # TODO: generate model
    realization = np.ones((real_nx+2*padding_x, real_ny+2*padding_y, real_nz+2*padding_z))*-1
    if hard_data is not None:
        if padding_z != 0:
            realization[padding_x:-padding_x, padding_y:-padding_y, padding_z:-padding_z] = hard_data
        else:
            realization[padding_x:-padding_x, padding_y:-padding_y, :] = hard_data
        if verbose:
            print('hard data is conditioned')
    x_0, x_1 = int(0 +padding_x), int(realization.shape[0] - padding_x)
    y_0, y_1 = int(0 +padding_y), int(realization.shape[1] - padding_y)
    z_0, z_1 = int(0 +padding_z), int(realization.shape[2] - padding_z)
    xx, yy, zz = np.meshgrid(range(x_0, x_1), range(y_0, y_1), range(z_0, z_1))
    random_path = np.array([i.flatten() for i in [xx, yy, zz]])x
    # TODO: random path without hard data
    return realization, [facies_ratio, unique_facies], [data_x, data_y, flag], random_path

def multi_points_modeling_multi_scaled(TI, n_level, level_size,
                                      template_size, 
                                      random_seed, 
                                      real_nx, real_ny, real_nz,
                                      hard_data = None,
                                      soft_data = None, 
                                      verbose = False):
    
    TI_s, grid_size_s = [], []
    nx, ny, nz = real_nx, real_ny, real_nz

    for level in range(n_level):
        TI_s.append(TI[::level_size**level, ::level_size**level, ::level_size**level])
        grid_size_s.append((nx, ny, nz))
        nx, ny, nz = round(nx/level_size), round(ny/level_size), round(nz/level_size)

    real_s = []
    if hard_data is None:
        real = np.ones(grid_size_s[-1]) * -1
    else:
        real = hard_data
    
    for idx, (level, TI_at_level, grid_size_at_level) in enumerate(zip(range(n_level)[::-1],TI_s[::-1], grid_size_s[::-1])):
        real = multi_points_modeling(TI_at_level, 
                                    template_size, 
                                    random_seed, 
                                    grid_size_at_level[0], grid_size_at_level[1], grid_size_at_level[2], 
                                    real, 
                                    soft_data,
                                    verbose)
        real_s.append(real)

        if level == 0:
            break
        real_next = np.ones(grid_size_s[level-1]) * -1
        real_next[1::level_size, 1::level_size, 1::level_size] = real
        real = real_next.copy()
    
    return real