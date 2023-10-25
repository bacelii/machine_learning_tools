
from functools import reduce
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist,squareform
import datetime
import itertools
import networkx as nx
import numpy as np
import time 
import trimesh
"""
Notes on functionality: 
np.concatenate: combines list of lists into one list like itertools does
np.ptp: gives range from maximum-minimum

np.diff #gets the differences between subsequent elements (turns n element --> n-1 elements)

np.insert(array,indexes of where you want insertion,what you want inserted before the places you specified) --> can do multiple insertions: 

Ex: 
x = np.array([1,4,5,10])
np.insert(x,slice(0,5),2)
>> output: array([ 2,  1,  2,  4,  2,  5,  2, 10])


If want to find the indexes of what is common between 2 1D arrray use
same_ids,x_ind,y_ind = np.intersect1d(soma_segment_id,connectivity_ids,return_indices=True)


"""
def compare_threshold(item1,item2,threshold=0.0001,print_flag=False):
    """
    Purpose: Function that will take a scalar or 2D array and subtract them
    if the distance between them is less than the specified threshold
    then consider equal
    
    Example: 
    nu = reload(nu)

    item1 = [[1,4,5,7],
             [1,4,5,7],
             [1,4,5,7]]
    item2 = [[1,4,5,8.00001],
            [1,4,5,7.00001],
            [1,4,5,7.00001]]

    # item1 = [1,4,5,7]
    # item2 = [1,4,5,9.0000001]

    print(nu.compare_threshold(item1,item2,print_flag=True))
    """
    item1 = np.array(item1)
    item2 = np.array(item2)

    if item1.ndim != item2.ndim:
        raise Exception(f"Dimension for item1.ndim ({item1.ndim}) does not equal item2.ndim ({item2.ndim})")
    if item1.ndim > 2 or item2.ndim > 2:
        raise Exception(f"compare_threshold does not handle items with greater than 2 dimensions: item1.ndim ({item1.ndim}), item2.ndim ({item2.ndim}) ")

    if item1.ndim < 2:
        difference = np.linalg.norm(item1-item2)
    else:
        difference = np.sum(np.linalg.norm(item1 - item2,axis=1))
    
    if print_flag:
        print(f"difference = {difference}")
        
    #compare against threshold and return result
    return difference <= threshold

def concatenate_lists(list_of_lists):
    try:
        return np.concatenate(list_of_lists)
    except:
        return []

def is_array_like(current_data,include_tuple=False):
    types_to_check = [type(np.ndarray([])),type(np.array([])),list,trimesh.caching.TrackedArray]
    if include_tuple:
        types_to_check.append(tuple)
    return type(current_data) in types_to_check

def non_empty_or_none(current_data):
    if current_data is None:
        return False
    else:
        if len(current_data) == 0:
            return False
        return True

def array_after_exclusion(
                        original_array=[],                    
                        exclusion_list=[],
                        n_elements=0):
    """
    To efficiently get the difference between 2 lists:
    
    original_list = [1,5,6,10,11]
    exclusion = [10,6]
    n_elements = 20

    array_after_exclusion(n_elements=n_elements,exclusion_list=exclusion)
    
    
    ** pretty much the same thing as : 
    np.setdiff1d(array1, array2)

    """
    
    
    if len(exclusion_list) == 0: 
        return original_array
    
    if len(original_array)==0:
        if n_elements > 0:
            original_array = np.arange(n_elements)
        else:
            raise Exceptino("No original array passed")
    else:
        original_array = np.array(original_array)
            
    mask = ~np.isin(original_array,exclusion_list)
    #print(f"mask = {mask}")
    return original_array[mask]

def load_dict(file_path):
    if file_path == type(Path()):
        file_path = str(file_path.absolute())
      
    my_dict = np.load(file_path,allow_pickle=True)
    return my_dict[my_dict.files[0]][()]

def get_coordinate_distance_matrix(coordinates):
    distance_matrix_condensed = pdist(coordinates,'euclidean')
    distance_matrix = squareform(distance_matrix_condensed)
    return distance_matrix

def get_matching_vertices(possible_vertices,ignore_diagonal=True,
                         equiv_distance=0,
                         print_flag=False):
    """
    ignore_diagonal is not implemented yet 
    """
    possible_vertices = possible_vertices.reshape(-1,3)
    
    dist_matrix = get_coordinate_distance_matrix(possible_vertices)
    
    dist_matrix_copy = dist_matrix.copy()
    dist_matrix_copy[np.eye(dist_matrix.shape[0]).astype("bool")] = np.inf
    if print_flag:
        print(f"The smallest distance (not including diagonal) = {np.min(dist_matrix_copy)}")
    
    matching_vertices = np.array(np.where(dist_matrix <= equiv_distance)).T
    if ignore_diagonal:
        left_side = matching_vertices[:,0]
        right_side = matching_vertices[:,1]

        result = matching_vertices[left_side != right_side]
    else:
        result = matching_vertices
        
    if len(result) > 0:
        return np.unique(np.sort(result,axis=1),axis=0)
    else:
        return result

def number_matching_vertices_between_lists(arr1,arr2,verbose=False):
    stacked_vertices = np.vstack([np.unique(arr1,axis=0),np.unique(arr2,axis=0)])
    stacked_vertices_unique = np.unique(stacked_vertices,axis=0)
    n_different = len(stacked_vertices) - len(stacked_vertices_unique)
    return n_different

def test_matching_vertices_in_lists(arr1,arr2,verbose=False):
    n_different = number_matching_vertices_between_lists(arr1,arr2)
    if verbose:
        print(f"Number of matching vertices = {n_different}")
    if n_different > 0:
        return True
    elif n_different == 0:
        return False
    else:
        raise Exception("More vertices in unique list")

"""
How can find pairwise distance:

example_skeleton = current_mesh_data[0]["branch_skeletons"][0]
ex_skeleton = example_skeleton.reshape(-1,3)


#sk.convert_skeleton_to_graph(ex_skeleton)

start_time = time.time()
distance_matrix = pdist(ex_skeleton,'euclidean')
print(f"Total time for pdist = {time.time() - start_time}")

returns a matrix that is a lower triangular matrix of size n*(n-1)/2
that gives the distance



"""
def find_matching_endpoints_row(branch_idx_to_endpoints,end_coordinates):
    match_1 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[0]).all(axis=1).reshape(-1,2)
    match_2 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[1]).all(axis=1).reshape(-1,2)
    return np.where(np.sum(match_1 + match_2,axis=1)>1)[0]

def matching_rows_old(vals,row,print_flag=False):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where((np.array(vals) == np.array(row)).all(axis=1))[0]

def matching_rows(vals,row,
                      print_flag=False,
                      equiv_distance = 0.0001):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    row = np.array(row).reshape(-1,3)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where(np.linalg.norm(vals-row,axis=1)<equiv_distance)[0]

def matching_row_index(vals,row):
    return matching_rows(vals,row)[0]


# ----------- made when developing the neuron class ------------- #
def sort_multidim_array_by_rows(edge_array,order_row_items=False,):
    """
    Purpose: To sort an array along the 0 axis where you maintain the row integrity
    (with possibly sorting the individual elements along a row)
    
    Example: On how to get sorted edges
    from datasci_tools import numpy_utils as nu
    nu = reload(nu)
    nu.sort_multidim_array_by_rows(limb_concept_network.edges(),order_row_items=True)
    
    """
    #print(f'edge_array = {edge_array} with type = {type(edge_array)}')
    
    #make sure it is an array
    edge_array = np.array(edge_array)
    
    #check that multidimensional
    if len(edge_array.shape ) < 2:
        print(f"edge_array = {edge_array}")
        raise Exception("array passed did not have at least 2 dimensions")
        
    #will rearrange the items to be in a row if not care about the order here
    if order_row_items:
        edge_array = np.sort(edge_array,axis=1)

    #sort by the x and then y of the egde
    def sorting_func(k):
        return [k[i] for i,v in enumerate(edge_array.shape)]

    #sorted_edge_array = np.array(sorted(edge_array , key=lambda k: [k[0], k[1]]))
    sorted_edge_array = np.array(sorted(edge_array , key=sorting_func))
    
    return sorted_edge_array



def sort_elements_in_every_row(current_array):
    return np.array([np.sort(yi) for yi in current_array])
# --------- Functions pulled from trimesh.grouping ---------- #

def sort_rows_by_column(array,column_idx,largest_to_smallest=True):
    """
    Will sort the rows based on the values of 1 column
    
    """
    order = array[:,column_idx].argsort()
    if largest_to_smallest:
        order = np.flip(order)
    return array[order]



def function_over_multi_lists(arrays,set_function):
    return reduce(set_function,arrays)

def setdiff1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.setdiff1d)

def intersect1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.intersect1d)

def intersect2d_multi_list(arrays):
    return function_over_multi_lists(arrays,nu.intersect2d)

def union1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.union1d)

def intersect1d(arr1,arr2,assume_unique=False,return_indices=False):
    """
    Will return the common elements from 2 possibly different sized arrays
    
    If select the return indices = True,
    will also return the indexes of the common elements
    
    
    """
    return np.intersect1d(arr1,arr2,
                         assume_unique=assume_unique,
                         return_indices=return_indices)

def setdiff1d(arr1,arr2,assume_unique=False,return_indices=True):
    """
    Purpose: To get the elements in arr1 that aren't in arr2
    and then to possibly return the indices of those that were
    unique in the first array
    
    
    
    """
    
    arr1 = np.array(arr1)
    leftout = np.setdiff1d(arr1,arr2,assume_unique=assume_unique)
    _, arr_1_indices, _ = np.intersect1d(arr1,leftout,return_indices=True)
    arr_1_indices_sorted= np.sort(arr_1_indices)
    if return_indices:
        return arr1[arr_1_indices_sorted],arr_1_indices_sorted
    else:
        return arr1[arr_1_indices_sorted]
    
def setdiff2d(arr1,arr2):
    try:
        return np.array([k for k in arr1 if len(nu.matching_rows(arr2,k))==0])
    except:
        return np.array([k for k in arr1 if len(nu.matching_rows_old(arr2,k))==0])
    
def intersect2d(arr1,arr2):
    try:
        return np.array([k for k in arr1 if len(nu.matching_rows(arr2,k))>0])
    except:
        return np.array([k for k in arr1 if len(nu.matching_rows_old(arr2,k))>0])
    
def divide_into_label_indexes(mapping):
    """
    Purpose: To take an array that attributes labels to indices
    and divide it into a list of the arrays that correspond to the indices of
    all of the labels
    
    """
    unique_labels = np.sort(np.unique(mapping))
    final_list = [np.where(mapping==lab)[0] for lab in unique_labels]
    return final_list

def turn_off_scientific_notation():
    np.set_printoptions(suppress=True)
    
def average_by_weights(values,weights):
    weights_normalized = weights/np.sum(weights)
    return np.sum(values*weights_normalized)

def angle_between_vectors(v1, v2, acute=True,degrees=True,verbose=False):
    """
    vec1 = np.array([0,0,1])
    vec2 = np.array([1,1,-0.1])
    angle(vec1,vec2,verbose=True)
    """

    dot_product = np.dot(v1, v2)
    if verbose:
        print(f"dot_product = {dot_product}")
    angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    if acute == True:
        rad_angle =  angle
    else:
        rad_angle =  2 * np.pi - angle
        
    if degrees:
        return  180* rad_angle/np.pi
    else:
        return rad_angle
            
    
    return return_angle


def intersecting_array_components(arrays,sort_components=True,verbose=False,perfect_match=False):
    """
    Purpose: 
    Will find the groups of arrays that are
    connected components based on overlap of elements
    
    Pseudocode: 
    1) Create an empty edges list
    2) Iterate through all combinations of arrays (skipping the redundants)
    a. heck if there is an intersection
    b. If yes then add to edges list
    3) Trun the edges into a graph 
    4) Return the connected components
    
    """
    
    array_edges = []
    for i,arr1 in enumerate(arrays):
        for j,arr2 in enumerate(arrays):
            if i < j:
                if perfect_match:
                    if len(arr1) != len(arr2):
                        continue
                intersect_elem = np.intersect1d(arr1,arr2)
                if perfect_match:
                    if len(intersect_elem) < len(arr1):
                        continue
                if len(intersect_elem)>0:
                    if verbose:
                        print(f"for edge {[i,j]}, # matching element = {len(intersect_elem)}")
                    array_edges.append([i,j])
                    
                    
    if verbose:
        print(f"array_edges = {array_edges}")
        
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(arrays)))
    G.add_edges_from(array_edges)
    
    conn_comps = list([list(k) for k in nx.connected_components(G)])
    
    if sort_components:
        conn_comps_lenghts = [len(k) for k in conn_comps]
        conn_comps_ordered = [conn_comps[k] for k in np.flip(np.argsort(conn_comps_lenghts))]
        if verbose: 
            print(f"Returning ordered connected components, original lens = {conn_comps_lenghts}")
        conn_comps =  conn_comps_ordered
    
    return np.array(conn_comps)

def array_split(array,n_groups):
    return np.array_split(array,n_groups)

def unique_rows(array):
    return np.unique(array,axis=0)

def unique_non_self_pairings(array):
    """
    Purpose: Will take a list of pairings and 
    then filter the list to only unique pairings where ther is no self
    pairing
    
    
    """
    array = np.array(array)
    
    if len(array) == 0 or (0 in array.shape) :
        return []
    
    array = np.unique(np.sort(np.array(array),axis=1),axis=0)
    array = array[array[:,0] != array[:,1]]
    return array


def choose_k_combinations(array,k):
    return list(itertools.combinations(array,k))
def all_unique_choose_2_combinations(array):
    """
    Given a list of numbers  or labels, will 
    determine all the possible unique pariings
    
    """
    starting_node_combinations = list(itertools.combinations(array,2))
    return nu.unique_non_self_pairings(starting_node_combinations)

def all_unique_choose_k_combinations(array,k):
    starting_node_combinations = list(itertools.combinations(array,k))
    return nu.unique_non_self_pairings(starting_node_combinations)
    

def unique_pairings_between_2_arrays(array1,array2):
    """
    Turns 2 seperate array into all possible comibnations of elements
    
    [1,2], [3,4]
    
    into 
    
    array([[1, 3],
       [1, 4],
       [2, 3],
       [2, 4]])
    
    
    """
    mesh = np.array(np.meshgrid(array1, array2))
    combinations = mesh.T.reshape(-1, 2)
    return combinations



def remove_indexes(arr1,arr2):
    return np.delete(arr1,arr2)

def mode_1d(array):
    return stats.mode(array)[0][0]
    
def all_subarrays (l): 
    """
    Ex: 
    from datasci_tools import numpy_utils as nu
    nu.all_subarrays([[1,"a"],[2,"b"],[3,"c"]])
    
    Output:
    [[],
     [[1, 'a']],
     [[2, 'b']],
     [[1, 'a'], [2, 'b']],
     [[3, 'c']],
     [[1, 'a'], [3, 'c']],
     [[2, 'b'], [3, 'c']],
     [[1, 'a'], [2, 'b'], [3, 'c']]]
    """
    base = []   
    lists = [base] 
    for i in range(len(l)): 
        orig = lists[:] 
        new = l[i] 
        for j in range(len(lists)): 
            lists[j] = lists[j] + [new] 
        lists = orig + lists 

    return lists

def random_2D_subarray(array,n_samples,
                      replace=False,
                      verbose=False):
    """
    Purpose: To chose a random number of rows from
    a 2D array
    
    Ex: 
    from datasci_tools import numpy_utils as nu
    import numpy as np

    y = np.array([[1,3],[3,2],[5,6]])
    nu.random_2D_subarray(array=y,
                      n_samples=2,
                      replace=False)
    """
    n_samples = int(n_samples)
    if verbose:
        print(f"Sampling {n_samples} rows from array of length {len(array)} with replacement = {replace}")
    random_indexes = np.random.choice(np.arange(len(array)),size=n_samples,replace=replace)
    return array[random_indexes]

def comma_str(num):
    return f"{num:,}"

def array_split(array,n_splits):
    """Split an array into multiple sub-arrays
    
    Ex: 
    from datasci_tools import numpy_utils as nu
    nu.array_split(np.arange(0,10),3)
    """
    return np.array_split(array,n_splits)


def repeat_vector_down_rows(array,n_repeat):
    """
    Ex: Turn [705895.1025, 711348.065 , 761467.87  ]
        into: 
    
    TrackedArray([[705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ]])
    """
    return np.repeat(array.reshape(-1,3),n_repeat,axis=0)

def all_partitions(array,
    min_partition_size = 2,
    verbose = False):
    """
    Will form all of the possible
    2 partions of an array
    where you can specify the minimum
    number of elements needed 
    for each possible partition
    
    Ex: 
    x = nu.all_partitions(array = np.array([4,5,6,9]))
    """

    choose_k_options = np.arange(min_partition_size,int(len(array)/2)+0.01).astype("int")
    if verbose: 
        print(f"choose_k_options = {choose_k_options}")

    array = np.array(array)

    all_partitions = []
    for k in choose_k_options:
        
        part_1 = nu.choose_k_combinations(array,k)
        part_2 = [np.setdiff1d(array,p1) for p1 in part_1]
        paired_partitions = [[list(u),list(v)] for u,v in zip(part_1,part_2)]

        
        
        if k == len(array)-k:
            paired_partitions = paired_partitions[:int(len(paired_partitions)/2)]

        if verbose:
            #print(f"part_1 = {part_1}")
            #print(f"part_2 = {part_2}")
            for j,pp in enumerate(paired_partitions):
                print(f"partition {j}: {pp}")
                
        all_partitions += paired_partitions

    return all_partitions

def float_to_datetime(fl):
    return datetime.datetime.fromtimestamp(fl)

def obj_array_to_dtype_array(array,dtype=None):
    return np.array(list(array),dtype=dtype)

def save_compressed(array,filepath):
    np.savez_compressed(filepath,data = array)
def load_compressed(filepath):
    return np.load(filepath)["data"]


# ---------- 6/7: Used for synapse filtering -------- #
def indices_of_comparison_func(func,array1,array2):
    """
    Returns the indices of the elements that result
    from applying func to array1 and array2
    """
    return np.nonzero(func(array1, array2))
def intersect_indices(array1,array2):
    """
    Returns the indices of the intersection of array1 and 2
    """
    return indices_of_comparison_func(np.in1d,array1,array2)

def polyval(poly,data):
    return np.polyval(poly,data)

def polyfit(x,y,degree):
    return np.polyfit(x, y, degree)

def weighted_average(array,weights):
    """
    Ex: 
    from datasci_tools import numpy_utils as nu
    nu.weighted_average(d_widths,d_sk_lengths)
    """
    return np.average(array,weights=weights)


def argnan(array):
    return np.where(np.isnan(np.array(array).astype(float)))[0]

def vector_from_endpoints(start_endpoint,end_endpoint,normalize_vector=True):
    vector = np.array(end_endpoint)-np.array(start_endpoint)
    if normalize_vector:
        vector = vector/np.linalg.norm(vector)
    return vector

def convert_to_array_like(array,include_tuple=False):
    """
    Will convert something to an array
    """
    if not nu.is_array_like(array,include_tuple=include_tuple):
        return [array]
    return array

def original_array_indices_of_elements(original_array,
                                      matching_array):
    """
    Purpose: Will find the indices of the matching array
    from the original array
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    nu.original_array_indices_of_elements(x,y)
    """
    return np.searchsorted(original_array,
                           matching_array)

def order_arrays_using_original_and_matching(original_array,
                                            matching_array,
                                            arrays,
                                            verbose = False,):
    """
    Purpose: To rearrange arrays so that 
    a specific array matches an original array

    Pseudocode: 
    1) Find the matching array elements
    2) For each array in arrays index using the matching indices
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    arrays  = [ np.array([ "hi","yes","but"])]
    arrays  = [ np.array([ "hi","yes","but"]), ["no","yes","hi"]]
    arrays  = [ np.array([ 1,2,3]), [7,8,9]]
    

    order_arrays_using_original_and_matching(original_array = x,
    matching_array = y,
    arrays=arrays,
    verbose = True)
    
    Return: 
    >>[array(['but', 'hi', 'yes'], dtype='<U3')]
    """
    mapping_indices = nu.original_array_indices_of_elements(original_array,
                                          matching_array)
    sorted_mapping_indices = np.argsort(mapping_indices)
    if verbose:
        print(f"mapping_indices = {mapping_indices}")
        print(f"sorted_mapping_indices = {sorted_mapping_indices}")

    reordered_arrays = [np.array(k)[sorted_mapping_indices] for k in arrays]
    reordered_arrays = [z if type(k) != list else list(z) for k,z in zip(arrays,reordered_arrays)]

    if verbose:
        print(f"reordered_arrays = {reordered_arrays}")

    return reordered_arrays

def order_array_using_original_and_matching(original_array,
                                            matching_array,
                                            array,
                                            verbose = False,):
    """
    Purpose: To rearrange arrays so that 
    a specific array matches an original array

    Pseudocode: 
    1) Find the matching array elements
    2) For each array in arrays index using the matching indices
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    arrays  = [ np.array([ "hi","yes","but"])]
    arrays  = [ np.array([ "hi","yes","but"]), ["no","yes","hi"]]
    arrays  = [ np.array([ 1,2,3]), [7,8,9]]
    

    order_arrays_using_original_and_matching(original_array = x,
    matching_array = y,
    arrays=arrays,
    verbose = True)
    
    Return: 
    >>[array(['but', 'hi', 'yes'], dtype='<U3')]
    """
    mapping_indices = nu.original_array_indices_of_elements(original_array,
                                          matching_array)
    sorted_mapping_indices = np.argsort(mapping_indices)
    if verbose:
        print(f"mapping_indices = {mapping_indices}")
        print(f"sorted_mapping_indices = {sorted_mapping_indices}")

    reordered_array = np.array(array)[sorted_mapping_indices]
    
    if type(array) == list:
        reordered_array = list(reordered_array)
        
    if verbose:
        print(f"reordered_array = {reordered_array}")

    return reordered_array

def divide_data_into_classes(classes_array,data_array,unique_classes=None):
    """
    Purpose: Will divide two parallel arrays of class and the data
    into a dictionary that keys to the unique class and hen 
    all of the data that belongs to that class
    """
    data_array = np.array(data_array)
    if unique_classes is None:
        unique_classes = np.unique(classes_array)
    
    return_dict = dict()
    for c in unique_classes:
        return_dict[c] = data_array[classes_array == c]
        
    return return_dict

def concatenate_arrays_along_last_axis_after_upgraded_to_at_least_2D(arrays):
    """
    Example: 
    from datasci_tools import numpy_utils as nu
    arrays = [np.array([1,2,3]), np.array([4,5,6])]
    nu.concatenate_arrays_along_last_axis_after_upgraded_to_at_least_2D(arrays)
    
    >> output:
    array([[1, 4],
       [2, 5],
       [3, 6]])
    """
    return np.c_[tuple(arrays)]

def min_max(array,axis=0):
    return np.min(array,axis=axis),np.max(array,axis=axis)
def min_max_3D_coordinates(array):
    return np.array(min_max(array,axis=0))

def bouning_box_corners(array):
    return min_max_3D_coordinates(array)

def bouning_box_midpoint(array):
    return np.mean(nu.bouning_box_corners(array),axis=0)

def bounding_box_side_lengths(array):
    min_max = nu.min_max(array)
    return min_max[1] - min_max[0]

def bounding_box_volume(array):
    return np.prod(nu.bounding_box_side_lengths(array))

def argsort_multidim_array_by_rows(array,descending=False):
    """
    Ex: 
    x = np.array([
        [2,2,3,4,5],
        [-2,2,3,4,5],
        [3,1,1,1,1],
        [1,10,10,10,10],
        [3,0,1,1,1],
        [-2,-3,3,4,5]
         ])
         
    #showing this argsort will correctly sort
    x[nu.argsort_multidim_arrays_by_rows(x)]
    
    >> Output: 
    
    array([[-2, -3,  3,  4,  5],
       [-2,  2,  3,  4,  5],
       [ 1, 10, 10, 10, 10],
       [ 2,  2,  3,  4,  5],
       [ 3,  0,  1,  1,  1],
       [ 3,  1,  1,  1,  1]])
         
    """
    unique_array = np.unique(array,axis=0)
    argsort_index = np.concatenate([nu.matching_rows_old(array,k) for k in unique_array])
    if descending:
        return argsort_index[::-1]
    else:
        return argsort_index

def matrix_of_row_idx(n_rows,n_cols=None):
    if n_cols is None:
        n_cols = n_rows
    return np.repeat(np.arange(0,n_rows).reshape(-1,n_rows).T,n_cols,axis=1)

def matrix_of_col_idx(n_rows,n_cols):
    return matrix_of_row_idx(n_cols,n_rows).T

def argsort_rows_of_2D_array_independently(array,descending=False):
    """
    Purpose: will return array for row idx and one for col idex
    that will sort the values of each row independently of the column
    
    Ex: 
    x = np.array([
        [2,2,3,4,5],
        [-2,2,3,4,5],
        [3,1,1,1,1],
        [1,10,10,10,10],
        [3,0,1,1,1],
        [-2,-3,3,4,5]
         ])
         
    row_idx,col_idx = nu.argsort_rows_of_2D_array_independently(x)
    x[row_idx,col_idx]
    
    Output:
    >>
    array([[ 2,  2,  3,  4,  5],
       [-2,  2,  3,  4,  5],
       [ 1,  1,  1,  1,  3],
       [ 1, 10, 10, 10, 10],
       [ 0,  1,  1,  1,  3],
       [-3, -2,  3,  4,  5]])
    """
    row_idx = nu.matrix_of_row_idx(*array.shape)
    col_idx = np.array([np.argsort(k)[::-1] if descending else np.argsort(k) for i,k in enumerate(array)])
    return row_idx,col_idx



def remove_nans(array):
    array = np.array(array)
    return array[~np.isnan(array)]

def all_directed_choose_2_combinations(array):
    """
    Ex: 
    seg_split_ids = ["864691136388279671_0",
                "864691135403726574_0",
                "864691136194013910_0"]
                
    output: 
    [['864691136388279671_0', '864691135403726574_0'],
     ['864691136388279671_0', '864691136194013910_0'],
     ['864691135403726574_0', '864691136388279671_0'],
     ['864691135403726574_0', '864691136194013910_0'],
     ['864691136194013910_0', '864691136388279671_0'],
     ['864691136194013910_0', '864691135403726574_0']]
    
    """
    combs = []
    for c1 in array:
        for c2 in array:
            if c1 == c2: 
                continue
            combs.append([c1,c2])
            
    return combs


def interpercentile_range(array,range_percentage,axis = None,verbose = False):
    """
    range_percentage should be 50 or 90 (not 0.5 or 0.9)
    
    Purpose: To compute the range that extends from
    (1-range_percentage)/2 to 0.5 + range_percentage/2
    
    Ex: 
    interpercentile_range(np.vstack([np.arange(1,11),
                                np.arange(1,11),
                                np.arange(1,11)]),90,verbose = True,axis = 1)
    """
    lower_perc = (100-range_percentage)/2
    upper_perc = 50 + range_percentage/2
    
    lower_n = np.percentile(array,lower_perc,axis=axis)
    upper_n = np.percentile(array,upper_perc,axis=axis)
    
    interpercentile_range = upper_n - lower_n
    
    if verbose:
        print(f"lower_n = {lower_n} (lower_perc = {lower_perc})")
        print(f"upper_n = {upper_n} (upper_perc = {upper_perc})")
        print(f"interpercentile_range = {interpercentile_range}")
        
    return interpercentile_range

def randomly_shuffle_array(array):
    return np.random.choice(array, len(array), replace=False)
def random_shuffled_indexes_for_array(array):
    idx_to_process = np.arange(0,len(array))
    return nu.randomly_shuffle_array(idx_to_process)


def all_choose_1_combinations_form_dict_values(parameter_dict,
                                        verbose = False):
    """
    Purpose: To generate a list of dictinoaries that 
    encompass all the possible parameter settings defined
    by the possible parameter settings in the dictionary
    
    Pseudocode: 
    
    
    """
    
    param_keys = list(parameter_dict.keys())
    total_list = [nu.convert_to_array_like(k) for k in parameter_dict.values()]
    all_param_comb = [p for p in itertools.product(*total_list)]
    if verbose:
        print(f"# of combinations = {len(all_param_comb)}")
    
    return [{k:v for k,v in zip(param_keys,l)} for l in all_param_comb]





from . import numpy_ml as nu