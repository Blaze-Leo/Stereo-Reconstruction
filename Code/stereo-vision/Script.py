# %% [markdown]
# Importing Libraries

# %%
import numpy as np
from tqdm import tqdm
import random
import copy
from PIL import Image
from collections import defaultdict
import pickle
import os


# %% [markdown]
# File Paths

# %%
# stereo_data_path = "/mnt/Personal/Projects/Depth_Reconstruction/Test_Folder/stereo_test/test_images/stereo.pkl"
# disparity_data_path = "/mnt/Personal/Projects/Depth_Reconstruction/Test_Folder/stereo_test/test_images/disparity.pkl"

stereo_data_path = "/kaggle/input/stereo-dataset-middlebury2014-input/stereo.pkl"
# disparity_data_path = "/kaggle/input/stereo-dataset-middlebury2014-input/disparity.pkl"
disparity_data_path = "/kaggle/working/disparity.pkl"

# %% [markdown]
# Accessory Functions

# %%


def load_stereo_data(pickle_path):
    """
    dict: The loaded dictionary with structure:
        {
            "folder1": {
                "order": int
                "im0": numpy array,
                "im1": numpy array,
                "calib": int
            },
            "folder2": { ... },
            ...
        }
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to unpickle file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading pickle file: {str(e)}")




# %%
def save_pickle(var, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
def rgb_to_mono(img_array):
    """Convert RGB image array to luminance (grayscale) using standard weights."""
    if len(img_array.shape) == 2:
        return img_array  # Already grayscale
    gray = np.dot(img_array[..., :3], [4, 4, 4])
    return gray.astype(np.int_)

# %%
def resize_image_array(image_array, scale_factor):
    # Convert array to PIL Image
    if len(image_array.shape) == 2:
        # Grayscale image
        img = Image.fromarray(image_array)
    elif len(image_array.shape) == 3:
        # RGB/RGBA image
        img = Image.fromarray(image_array.astype('uint8'))
    else:
        raise ValueError("Input array must be 2D (grayscale) or 3D (color)")
    
    # Calculate new dimensions
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize using Lanczos resampling (high quality)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert back to numpy array
    resized_array = np.array(resized_img)
    
    # Preserve original dtype for grayscale
    if len(image_array.shape) == 2:
        resized_array = resized_array.astype(image_array.dtype)
    
    return resized_array

# %% [markdown]
# Texture Segmentation

# %%

def texture_segmentation(counter, image, threshold, disable=False):
    
    h, w = image.shape
    output = np.zeros_like(image, dtype=int)
    texture_label = 1
    texture_dict = defaultdict(list)
    
    # Create list of all possible coordinates
    all_coords = [(y, x) for y in range(h) for x in range(w)]
    random.shuffle(all_coords)  # Randomize processing order
    
    with tqdm(total=h*w, desc=f"(Order - {counter})Segmenting Textures", disable=disable) as pbar:
        for y, x in all_coords:
            if output[y, x] != 0:
                pbar.update(1)
                continue
                
            org_value = image[y, x]
            stack = [(y, x)]
            pixels_processed = 0
            
            # Perform flood fill
            while stack:
                cy, cx = stack.pop()
                if output[cy, cx] != 0:
                    continue
                    
                output[cy, cx] = texture_label
                texture_dict[texture_label].append((cy, cx))
                pixels_processed += 1
                
                # Check 4-connected neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and 
                        output[ny, nx] == 0 and 
                        abs(int(image[ny, nx]) - int(org_value)) <= threshold):
                        stack.append((ny, nx))
            
            texture_label += 1
            pbar.update(pixels_processed)
    
    return output, texture_dict

# %% [markdown]
# Generating Disparity 

# %%

def generate_disparity(file_name, counter, image_l, image_r, text_dict, search_range,disable=False):
    
    # Validate inputs
    assert image_l.shape == image_r.shape, "Images must have the same shape"
    h, w = image_l.shape
    disparity = np.zeros((h, w), dtype=int)
    
    # Pre-compute all possible right window shifts
    shifts = np.arange(search_range)
    
    for tex in tqdm(text_dict, desc=f"(Order - {counter} )Generating Disparity for {file_name}",disable=disable):
        coords = np.array(text_dict[tex])
        i, j = coords[:, 0], coords[:, 1]
        
        # Compute all possible right windows at once
        j_shifted = j.reshape(-1, 1) - shifts.reshape(1, -1)
        
        # Mask for valid coordinates
        valid_mask = (j_shifted >= 0) & (j_shifted < w)
        all_valid = valid_mask.all(axis=0)
        
        # Initialize SAD values with infinity (for invalid shifts)
        sad_values = np.full(search_range, np.inf)
        
        # Compute SAD only for valid shifts
        for k in np.where(all_valid)[0]:
            right_j = j - k
            # Vectorized SAD computation
            diff = image_l[coords[:, 0], coords[:, 1]] - image_r[coords[:, 0], right_j]
            sad_values[k] = np.mean(np.abs(diff))
        
        # Find best disparity (minimum SAD)
        if not np.all(np.isinf(sad_values)):
            best_disparity = np.argmin(sad_values)
            disparity[coords[:, 0], coords[:, 1]] = best_disparity
    
    return disparity

# %% [markdown]
# Post Processing

# %%
import numpy as np
from collections import deque

def post_process_texture(disp_matrix, texture, threshold_percent):
    
    if disp_matrix.shape != texture.shape:
        raise ValueError("disp_matrix and texture must have the same shape")
    
    rows, cols = disp_matrix.shape
    processed = np.zeros_like(texture, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Create a view for faster access
    texture_flat = texture.ravel()
    disp_flat = disp_matrix.ravel()
    processed_flat = processed.ravel()
    
    for idx in range(len(texture_flat)):
        if processed_flat[idx]:
            continue
            
        current_texture = texture_flat[idx]
        queue = deque([idx])
        region_indices = []
        
        # BFS using flat indices
        while queue:
            flat_idx = queue.popleft()
            if not processed_flat[flat_idx]:
                processed_flat[flat_idx] = True
                region_indices.append(flat_idx)
                
                # Convert to 2D coordinates for neighbor checking
                x, y = np.unravel_index(flat_idx, (rows, cols))
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        neighbor_idx = np.ravel_multi_index((nx, ny), (rows, cols))
                        if (not processed_flat[neighbor_idx] and 
                            texture_flat[neighbor_idx] == current_texture):
                            queue.append(neighbor_idx) # type: ignore
        
        # Process the region
        if region_indices:
            region_disps = disp_flat[region_indices]
            median_disp = np.median(region_disps)
            threshold = median_disp * threshold_percent
            
            # Vectorized outlier detection and replacement
            outliers = np.abs(region_disps - median_disp) > threshold
            disp_flat[region_indices] = np.where(outliers, median_disp, region_disps)
    
    return disp_matrix

# %% [markdown]
# Execution

# %%
def single_runner(file_name, stereo_dict,resize):
    
    image_array_l = stereo_dict["im0"]
    image_array_r = stereo_dict["im1"]
    calib_dim = stereo_dict["calib"]
    
    
    resize_scale = resize

    rgb_l = resize_image_array(image_array_l,resize_scale)
    rgb_r = resize_image_array(image_array_r,resize_scale)

    # maximum colour density is now 1020

    gray_array_l = rgb_to_mono(rgb_l)
    gray_array_r = rgb_to_mono(rgb_r)


    # print("Image Shape - ",gray_array_l.shape)
    # print("Total Pixels - ",(gray_array_l.shape[0]*gray_array_l.shape[1]))

    texture,texture_dict = texture_segmentation(stereo_dict["order"], gray_array_l,20,disable=False)

    search = round(calib_dim*resize_scale)

    disp_matrix = generate_disparity(file_name, stereo_dict["order"], gray_array_l,gray_array_r,texture_dict,search,disable=False)

    # print("Search Distance - ",search)

    smoothed = copy.deepcopy(disp_matrix)

    post_process_iteration = 4

    for _ in tqdm(range(post_process_iteration), desc="Post Processing",disable=True):
        texture,_ = texture_segmentation(stereo_dict["order"],gray_array_l,100,disable=True)
        smoothed = post_process_texture(smoothed,texture, 0.07)

    print("Completed Running - ",file_name)
        
    return smoothed

    # plot_viridis_matrix(disp_matrix)
    # plot_viridis_matrix(smoothed)


# %%
import multiprocessing
from tqdm import tqdm
from collections import defaultdict

def process_wrapper(args):
    folder, data, resize = args
    multiprocessing.current_process().name = f"Processing {folder}"
    return folder, single_runner(folder, data, resize)

def parallel_process_stereo_data(stereo_data, resize, num_processes=None):

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    disparity_dict = defaultdict(dict)
    
    # Prepare arguments for multiprocessing
    args = [(folder, stereo_data[folder], resize) for folder in stereo_data]
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes, 
                            initializer=tqdm.set_lock, 
                            initargs=(multiprocessing.RLock(),)) as pool:
        
        # Use imap_unordered for faster processing with progress updates
        results = list(tqdm(pool.imap_unordered(process_wrapper, args),
                      total=len(stereo_data),
                      desc="Executing Files"))
        
        # Collect results
        for folder, result in results:
            disparity_dict[folder] = result # type: ignore
    
    return dict(disparity_dict)



# %%

stereo_data_org = load_stereo_data(stereo_data_path)
stereo_data={}

key_set=[]

for key in stereo_data_org:
    key_set.append(key)
    
key_set=key_set[:50]

for key in key_set:
    stereo_data[key]= stereo_data_org[key]
    


disparity_dict={}

resize = 1


disparity_dict = parallel_process_stereo_data(stereo_data, resize)
    
save_pickle = save_pickle(disparity_dict,disparity_data_path) # type: ignore


# %%
# stereo_data = load_stereo_data(stereo_data_path)

# disparity_dict={}

# resize = 0.0625


# for folder in tqdm(stereo_data,desc="Executing Files"):
    
#     disparity_dict[folder] = single_runner(stereo_data[folder],resize)
    
# save_pickle = save_pickle(disparity_dict,disparity_data_path)



