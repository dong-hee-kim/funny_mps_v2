import numpy as np
import itertools

def curate_training_image(TI_3D: np.ndarray, template_size: list[int], percentage_2_use: float = 0.8, cross = True):
    # define the training image size and template size
    """
    Curates a training image by extracting a tabular form dataset based on the provided template size.
    
    The function processes a 3D training image (TI) to create a dataset suitable for training 
    machine learning models. It extracts sub-volumes from the image according to the specified 
    template size and arranges them into a tabular format. The central element of each sub-volume 
    serves as the target output (data_y), while the surrounding elements serve as the input features (data_x).
    Optionally, a subset of the data can be used according to the percentage_2_use parameter.

    Parameters:
    TI (np.ndarray): The 3D training image.
    template_size (list[int]): The dimensions of the template used for extracting sub-volumes.
    percentage_2_use (float, optional): The fraction of data to use. Defaults to 0.8.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the input features (data_x) and the target outputs (data_y).
    """

    TI_x, TI_y, TI_z = TI_3D.shape
    padding_x, padding_y, padding_z = int((template_size[0]-1)/2), int((template_size[1]-1)/2), int((template_size[2]-1)/2)
    
    # extract the training image to make a tabular form data
    x_0, x_1 = int(0 +padding_x), int(TI_x - padding_x)
    y_0, y_1 = int(0 +padding_y), int(TI_y - padding_y)
    z_0, z_1 = int(0 +padding_z), int(TI_z - padding_z)
    
    template_size_x, template_size_y, template_size_z = (x_1 - x_0), (y_1 - y_0), (z_1 - z_0)
    data = np.zeros((np.prod([template_size_x, template_size_y, template_size_z]), np.prod(template_size)))
    for zi, z  in enumerate(range(z_0, z_1)):
        for yi, y  in enumerate(range(y_0, y_1)):
            for xi, x in enumerate(range(x_0, x_1)):
                for i, [tx, ty, tz] in enumerate(itertools.product(range(template_size[0]),
                                                                range(template_size[1]),
                                                                range(template_size[2]))):
                    
                    data[xi + yi*template_size_x + zi*template_size_x*template_size_y, i] = TI_3D[x+(tx-padding_x), y+(ty-padding_y), z+(tz-padding_z)]

    # train some ML model the above tabular data
    center_index = int((np.prod(template_size)-1)/2)
    indexes = np.arange(np.prod(template_size))
    if cross:
        ind_x, ind_y, ind_z = indexes%template_size[0], indexes//(template_size[0]), np.zeros_like(indexes)
        ind_x_center, ind_y_center, ind_z_center = ind_x[center_index], ind_y[center_index], ind_z[center_index]
        dist_from_center = [sum([abs(ind_x[i]-ind_x_center), abs(ind_y[i]-ind_y_center), abs(ind_z[i]-ind_z_center)]) for i in indexes]
        flag = [i for i in indexes if (i != center_index) & (dist_from_center[i]<=template_size[0]//2)]
        data_x = data[:, flag].reshape(-1, int((np.prod(template_size)-1)/2)).astype(np.int16)
    else:
        flag = [i for i in indexes if i != center_index]
        data_x = data[:, flag].reshape(-1, np.prod(template_size)-1).astype(np.int16)
    data_y = data[:, center_index].reshape(-1, 1).astype(np.int16)

    # in case of using partial data
    if percentage_2_use < 1:
        mask = np.random.choice(np.arange(data_x.shape[0]), int(percentage_2_use*data_x.shape[0]), replace=False)
        data_x = data_x[mask]
        data_y = data_y[mask]

    return data_x, data_y, flag
