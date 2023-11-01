# load the libraries

import numpy as np
import os
import sys
from collections import Counter
import pandas as pd

import scipy.io as sio
import cv2

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, get_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from skimage import measure

from shapely import Polygon, distance
sys.path.append('../')

colors = {
    1 : np.array([1, 0.5, 0]), # orange Neutrophil
    2 : np.array([0.2, 0.8, 0.2]), # epithelial (any)
    3 : np.array([1, 0, 0]), # red Lymphocyte
    4 : np.array([0, 1, 1]), # cyan Plasma
    5 : np.array([0, 0, 1]), # Blue Eosinophil
    6 : np.array([1, 1, 0]), # yellow Connective
    7 : np.array([0, 0, 0]), # black epithelial (malignant)
    8 : np.array([0, 0, 0])} # epithelial (healthy)

legend_short = {
    1 : 'Neutrophil',
    2 : 'Epithelial',
    3 : 'Lymphocyte',
    4 : 'Plasma',
    5 : 'Eosinophil',
    6 : 'Connective'}

legend_long = {
    1 : 'Neutrophil',
    2 : 'Epithelial (any)',
    3 : 'Lymphocyte',
    4 : 'Plasma',
    5 : 'Eosinophil',
    6 : 'Connective',
    7 : 'Epithelial (mali)',
    8 : 'Epithelial (heal)'}

default_eval_metrics = {'AUROC': 'roc_auc', 'avg_precision': 'average_precision', 'Accuracy': make_scorer(accuracy_score), 'F1-score':'f1'}

def inst_to_class(inst:np.ndarray, types:np.ndarray, to_keep:list=list(legend_short.keys())) -> np.ndarray:
    to_del = [i for i in range(7)].pop(to_keep)
    types = types.flatten()
    new = np.zeros(inst.shape)

    for cell_type in to_del:
        types[types==cell_type] = 0
    
    for value in np.unique(types):
        pos = list(np.where(types==value)[0]+1)
        bool_map = np.isin(inst,pos)
        new += bool_map * value
    
    return new

def gray_to_color(mask:np.ndarray, color_map:dict) -> np.ndarray:

    new = np.zeros(mask.shape + (3,))
    for i in range(3):
        for key in color_map.keys():
            new[...,i] += (mask==key) * color_map[key][i]
    return new

def get_cell_count(result_mat:dict, legend:dict=legend_short, class_field:str='class') -> dict:

    cls = set(legend.keys())
    numbers = Counter(result_mat[class_field].flatten())
    
    for key in cls - set(numbers.keys()):
        numbers[key] = 0

    cell_count = numbers
    
    return cell_count

def get_key(dictionary, val):
   
    for key, value in dictionary.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def pearson_correlation(df, target='malignant', plot_heatmap=True, num_only=True, high_only:bool=False, threshold:float=0.4):
    """
    This function calculates the pearson correlation between the features and the target variable

    Args:
        df: dataframe
        plot_heatmap: boolean, if True, a heatmap of the correlation is plotted
    Returns:
        corr: dataframe, sorted by the correlation with the target variable
    """
    corr = df.corr('pearson', numeric_only=num_only)[[target]].sort_values(by=target, ascending=False)
    if plot_heatmap:
        if high_only:
            corr_plot = corr.iloc[np.array(np.abs(corr)>threshold),:]
        else:
            corr_plot = corr
        fig,(ax) = plt.subplots(figsize=(4,len(corr_plot)/4), ncols=1)
        sns.heatmap(corr_plot,ax=ax, annot=True, vmin= -1, vmax = 1, center = 0, cmap = 'coolwarm')
        plt.title('Pearson Correlation Heatmap of continuous variables')
        plt.tight_layout()
        plt.show()
        return corr, corr_plot

    return corr

def pairplot(df, target = 'malignant', features = None):
    """
    This function plots a pairplot of the features

    Args:
        df: dataframe
        target: string, name of the target variable
        features: list of strings, names of the features. If None, all features are plotted
    """
    if features is None:
        features = pearson_correlation(df,plot_heatmap=False).index.tolist()
    # exclude target variable
    features = [feature for feature in features if feature != target]
    # exclude categorical features
    features = [feature for feature in features if df[feature].dtype != 'object']
    sns.pairplot(df, hue=target, vars=features)
    plt.show()
    return None


def cramers_v(contigency):
    """
    This function calculates the Cramer's V statistic for categorical-categorical association.    

    Args:
        contigency: numpy array, contigency table
    Returns:
        cramers_v: float, Cramer's V statistic
    """
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(contigency, correction=False)
    N = np.sum(np.array(contigency))
    minimum_dimension = min(np.array(contigency).shape)-1
    cramers_v = np.sqrt((chi2/N) / minimum_dimension)
    return cramers_v

def plot_contingency(df, target = 'malignant', subplot_size=(1, 6), figsize=(17, 3)):
    """
    This function plots the contingency tables of the categorical features 
    and reports the Cramer's V statistic

    Args:
        df: dataframe
        target: string, name of the target variable
    """
    fig, axs = plt.subplots(subplot_size[0], subplot_size[1], figsize=figsize)
    # first compute the cramers v for all and then plot the contingency tables in descending order
    cramers_v_list = []
    for col in df.select_dtypes(include='object').columns:
        contigency = pd.crosstab(df[col], df[target])
        cramers_v_list.append(cramers_v(np.array(contigency)))
    for i, col in enumerate(df.select_dtypes(include='object').columns[np.argsort(cramers_v_list)[::-1]]):
        contigency = pd.crosstab(df[col], df[target])
        sns.heatmap(contigency, annot=True, ax=axs.flatten()[i], cmap='YlGnBu', fmt=".0f", cbar = False)
        axs.flatten()[i].set_title(f'{col}\nCramer\'s V: {cramers_v(np.array(contigency)):.2f}')
        axs.flatten()[i].set(ylabel="")
    # set remaining subplots to invisible
    for i in range(len(df.select_dtypes(include='object').columns), subplot_size[0]*subplot_size[1]):
        axs.flatten()[i].axis('off')
    plt.show()
    return None

def get_data_path_from_img(img_file:str)->str:
    data_path = os.path.normpath(img_file).split(os.path.sep)
    data_path =  "/".join([data_path[i] for i in range(len(data_path)-2)])
    return data_path

def get_result_mat_from_img(img_file:str, label_folder:str) -> str:
    
    data_path = get_data_path_from_img(img_file)
    return os.path.join(os.path.join(data_path, label_folder), os.path.splitext(os.path.basename(img_file))[0]+'.mat')

def load_result_mat(img_file:str, label_folder:str='Labels') -> dict:
    result_mat = get_result_mat_from_img(img_file, label_folder)
    return sio.loadmat(result_mat)

def show_tile_labeled(img_file:str, size_lim:int=600, marker_size:int=7, legend:dict=legend_short) -> None:
    
    size_lim = [size_lim, size_lim]
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    result_mat = load_result_mat(img_file)
    # result_mat = os.path.join(os.path.join(data_path,'Labels'), os.path.basename(img_file).split('.')[0]+'.mat')
    
    image = cv2.imread(img_file)
    # result_mat = sio.loadmat(result_mat)
    inst_map = result_mat['inst_map'] 
    numbers = get_cell_count(result_mat, legend)

    print('instance map shape', inst_map.shape)
    # double check the number of instances is the same as the number of type predictions
    print('number of instances', len(np.unique(inst_map)[1:].tolist()))
    print('number of type predictions', len(np.unique(result_mat["class"])))
    # print('overlay shape', overlay.shape)
    print(f'Number of instance from each type:')
    for i in sorted(numbers.keys()):
        print(f'{i}.- {legend[i]}: {numbers[i]} ({100*numbers[i] / len(np.unique(inst_map)[1:].tolist()):.2f}%)')

    axs[0].imshow(image[:size_lim[0], :size_lim[1]])
    axs[0].axis('off')
    axs[0].set_title(f'Image {os.path.basename(img_file)}')

    for i, inst in enumerate(result_mat['centroid']):
        if inst[0]<size_lim[0] and inst[1]<size_lim[1]:
            cls = result_mat["class"][i][0]
            axs[0].scatter(x=inst[0], y=inst[1], color=colors[cls], s=marker_size)

    els = [Line2D([0],[0], marker='o', color='w', markerfacecolor= colors[i], label=legend[i], markersize=marker_size) for i in colors.keys()]
    axs[0].legend(handles=els, loc='best')

    axs[1].imshow(gray_to_color(inst_to_class(inst_map[:size_lim[0], :size_lim[1]], result_mat["class"]), colors))
    axs[1].axis('off')
    axs[1].set_title('Instance Segmentation Masks')
    plt.show()

    return None


def build_df(data_path:str, legend:dict=legend_short, labels_folder:str='Labels', class_field:str='class') -> pd.DataFrame:
    df = pd.DataFrame(columns=['file','cohort', 'img_dim_x', 'img_dim_y','malignant', 'total_count', 'neut_count', 'epit_count', 'lymp_count', 'plas_count', 'eosi_count', 'conn_count'])

    for path, __, files in os.walk(data_path):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')) and "images_40x" not in os.path.join(path, name):
                img_file = os.path.join(path, name)
                result_mat = load_result_mat(img_file, labels_folder)

                cohort = os.path.basename(img_file).split('_')[0] if 'TCGA' not in img_file else 'TCGA'
                size = cv2.imread(img_file).shape

                count_by_type = get_cell_count(result_mat, legend, class_field)
                total = sum(count_by_type.values())
                df.loc[len(df),:] = [img_file,
                                    cohort,
                                    size[0],
                                    size[1],
                                    1 if "malignant" in img_file else 0,
                                    total,
                                    count_by_type[get_key(legend, "Neutrophil")] / total,
                                    count_by_type[get_key(legend, "Epithelial")] / total,
                                    count_by_type[get_key(legend, "Lymphocyte")] / total,
                                    count_by_type[get_key(legend, "Plasma")] / total,
                                    count_by_type[get_key(legend, "Eosinophil")] / total,
                                    count_by_type[get_key(legend, "Connective")] / total,
                                    ]
    return df.infer_objects()


def dataset_histogram(im_files:list, ax, get_mean:bool=False, get_median:bool=False, bins:int=10) -> np.ndarray:
      
      vals= {"r": [], "g": [], "b": []}

      for file in im_files:
            im = cv2.imread(file)
            vals["r"].extend(im[...,0].flatten())
            vals["g"].extend(im[...,1].flatten())
            vals["b"].extend(im[...,2].flatten())

      hist_r, bins_r = np.histogram(vals["r"], bins=bins)
      hist_g, bins_g = np.histogram(vals["g"], bins=bins)
      hist_b, bins_b = np.histogram(vals["b"], bins=bins)

      if get_mean:
            mean_val = np.array([np.mean(vals["r"]),np.mean(vals["g"]),np.mean(vals["b"])])
      elif get_median:
            median_val = np.array([np.median(vals["r"]),np.median(vals["g"]),np.median(vals["b"])])
      del vals
      
      ax.stairs(hist_r, bins_r, fill=False, color='r', label="Red")
      ax.stairs(hist_g, bins_g, fill=False, color='g', label="Green")
      ax.stairs(hist_b, bins_b, fill=False, color='b', label="Blue")

      if get_mean:
            return mean_val/255
      elif get_median:
            return median_val/255
      

def get_mean_image(im_file:str, ignore_white:bool=False, white_thres:float=240., ignore_black:bool=False, black_thres:float=15.) -> np.ndarray:
    im = cv2.imread(im_file)

    if ignore_white:
        if np.max(im)<=1:
            white_thres = white_thres/255.
        white = np.prod(im > white_thres, axis=2)
        keep = (white != 1) *1.
        keep[keep==0] = np.nan
        im_no_white = np.stack([keep,keep,keep], axis=2) * im
        return np.nanmean(im_no_white, axis=(0,1)), np.nanstd(im_no_white, axis=(0,1))
    elif ignore_black:
        if np.max(im)<=1:
            black_thres = black_thres/255.
        black = np.prod(im < black_thres, axis=2)
        keep = (black != 1) *1.
        keep[keep==0] = np.nan
        im_no_black = np.stack([keep]*3, axis=-1) * im
        return np.nanmean(im_no_black, axis=(0,1)), np.nanstd(im_no_black, axis=(0,1))

    else:
        return np.mean(im, axis=(0,1)), np.std(im, axis=(0,1))




def evaluate_trained_clf(clf, x_test, y_test, scoring:dict=default_eval_metrics):
    # Helper function to compute evaluation metrics for trained classifiers

    if len(np.unique(y_test)) <= 2:

        for score, score_fn in scoring.items():
            # Fix up function reference
            score_fn = get_scorer(score_fn) if type(score_fn) == str else score_fn
            print("%s: %f" % (score, score_fn(clf, x_test, y_test)))
    print(confusion_matrix(y_test, clf.predict(x_test)))


def get_features_and_target(df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, target:str, scaling:str='standard'):

    def separate_xy(df, target, nan_to:str='zero'):
        df = df.fillna(0)
        x = df.loc[:, df.columns[df.columns != target].tolist()].to_numpy()
        y = df[target].to_numpy()
        return x, y

    if scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    train_x , train_y = separate_xy(df_train, target)
    test_x , test_y = separate_xy(df_test, target)
    val_x, val_y = separate_xy(df_val, target)

    if scaler is not None:
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)

    return train_x, val_x, test_x, train_y, val_y, test_y
    

def get_feature_columns(df:pd.DataFrame, target:str = "HeartDisease"):
    features = [feature for feature in df.columns if feature != target]
    return features

def fit_and_evaluate_logistic_model(df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, penalty='l1', 
                                    solver='liblinear', C=1.0, scaling:str=None, target:str='malignant'):
    """
    This function fits a logistic logistic regression model, 
    prints the weights obtained and evaluate the results in the test set.    

    Args:
        df_train: dataframe with the train dataset
        df_test: dataframe with the test dataset
        penalty: penalty for the logistic model {'l1', 'l2', 'elasticnet', None} (only used if group=False)
        solver: solver used to fit the model {'liblinear', 'saga'} (only used if group=False)
        C: inverse of regularization strength
        group: boolean indicating if it is a group lasso model
        groups: list indicating the partition of features used in group lasso (only used if group=True)
    Returns:
        logistic_clf: logistic regression model
    """

    logistic_clf = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=1000)

    X_train, X_val,X_test, y_train, y_val, y_test = get_features_and_target(df_train, df_val, df_test, target, scaling)

    logistic_clf.fit(X_train, y_train)

    print('COEFFICIENTS OBTAINED BY THE LOGISTIC MODEL')
    coefs = logistic_clf.coef_[0]
    intercept = logistic_clf.intercept_[0]
    for (name, weight) in zip(get_feature_columns(df_train, target=target), coefs):
        print(name, weight)
    print('Intercept', intercept)

    print('\nRESULTS OBTAINED IN THE VAL SET')
    eval_metrics = default_eval_metrics
    evaluate_trained_clf(logistic_clf, X_val, y_val, scoring=eval_metrics)

    print('\nRESULTS OBTAINED IN THE TEST SET')
    evaluate_trained_clf(logistic_clf, X_test, y_test, scoring=eval_metrics)

    return logistic_clf


def plot_logistic_feature_weights(logistic_clf, df, target:str='malignant', n:int=25):
    """
    This function plots the coefficients of a fitted classifier  

    Args:
        logistic_clf: fitted logistic regression model
        df: dataframe used to get the feature names
        group: boolean indicating if it is a group lasso model
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    coefs = logistic_clf.coef_[0]
    feats = get_feature_columns(df, target=target)
    indices = np.argsort(np.abs(coefs))

    feats = [feats[i] for i in indices[-n:]]  
    coefs = [coefs[i] for i in indices[-n:]]
    
    n = n if n < len(coefs) else len(coefs)
    sns.barplot(x=feats, y=coefs)
    ax.set_title("Logistic regression feature coefficients")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    plt.show()

def prepare_box_for_contours(box, shape, pad=3):
    """Marginally pads a bounding box so that object boundaries
    are not on cropped image patch edges.
    """
    box = [box[0], box[2], box[1], box[3]]
    for i in range(2):
        box[i] = min(0, box[i] - pad)
        box[i+2] = max(shape[i], box[i] + pad)
        
    slices = tuple([slice(box[i], box[i+2]) for i in range(2)])
    top_left = np.array(box[:2])[None] # (1, 2)
    return slices, top_left

def make_polygons_from_mask(mask,bbox):
    """Constructs a polygon for each object in a mask. Returns
    a dict where each key is a label id and values are shapely polygons.
    """
    polygons = {}
    for i, rp in enumerate(bbox):
        # Faster to compute contours on small cell tiles than the whole image
        box_slices, box_top_left = prepare_box_for_contours(list(rp), mask.shape, pad=0)
        label_mask = mask[box_slices] == i + 1
        contours = measure.find_contours(label_mask)
        label_cnts = np.concatenate(contours, axis=0)

        polygons[i + 1] = Polygon(label_cnts + box_top_left)
    
    return polygons

def pairwise_polygon_distance(polygons_dict):
    """Computes pairwise distance between all polygons in
    a dictionary. Returns a dictionary of distances.
    """
    distances = {l: {} for l in polygons_dict.keys()}
    for i in polygons_dict.keys():
        for j in polygons_dict.keys():
            # nested loop is slow but we cache results
            # to eliminate duplicate work
            if i != j and distances[i].get(j) is None:
                distances[i][j] = distance(polygons_dict[i], polygons_dict[j])
                
    return distances

def get_lowest_values_dict(parent_dict:dict, n):
    # sort the dictionary by values and keep the first 10 items
    sorted_items = sorted(parent_dict.items(), key=lambda x: x[1])[:n]
    # create a new dictionary from the sorted items
    lowest_dict = dict(sorted_items)
    return lowest_dict

def get_nn_distance(key, distances_dict, n):
    """Returns the nearest neighbor(s) for a polygon
    along with the distance.
    """
    return get_lowest_values_dict(parent_dict=distances_dict[key], n=n)