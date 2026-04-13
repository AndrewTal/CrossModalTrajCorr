import os
import json
import warnings
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import geopandas as gpd    
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr, kendalltau, wasserstein_distance
from scipy.spatial.distance import cdist
from matplotlib import cm
from shapely.geometry import Point, Polygon, MultiPolygon, shape as shapely_shape
from shapely.ops import unary_union
from tqdm import tqdm
from sklearn.gaussian_process.kernels import RBF
from scipy.special import comb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

plt.switch_backend('Agg')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['font.size'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28

# ---------------------------- Parameter configuration ----------------------------
#Sample Data Directory
ROOT_DIR ="samples"  
#Result Saving Directory
SAVE_ROOT ="results/GigaPath_fix"
os.makedirs(SAVE_ROOT, exist_ok=True)

MIN_SPOTS = 100  
N_PCS = 30  
HVG_N = None
RESOLUTION = 0.3
DIFFMAP = True
  
# Name of Image Feature
FEATURE_KEY = "GigaPath_features"
BARCODE_KEY = "barcode"
CELLVIT_DIRNAME = "cellvit_seg"
CELLVIT_FILENAME_TEMPLATE = "{sample}_cellvit_seg.geojson"
IROOT_STRATEGY = 'no_tumor_else_min'

# ----------------------------Obtain the spatial coordinates of the spot ----------------------------
def get_spatial_coords(adata, sample_name):

    if "pxl_col_in_fullres" in adata.obs.columns and "pxl_row_in_fullres" in adata.obs.columns:
        x_coords = adata.obs["pxl_col_in_fullres"].astype(float).values
        y_coords = adata.obs["pxl_row_in_fullres"].astype(float).values
        valid_mask = ~(np.isnan(x_coords) | np.isinf(x_coords) | (x_coords <= 0) |
                      np.isnan(y_coords) | np.isinf(y_coords) | (y_coords <= 0))
        return x_coords, y_coords, valid_mask
    
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[0] == adata.n_obs and adata.obsm["spatial"].shape[1] >= 2:
        x_coords = adata.obsm["spatial"][:, 0].astype(float)
        y_coords = adata.obsm["spatial"][:, 1].astype(float)
        valid_mask = ~(np.isnan(x_coords) | np.isinf(x_coords) | (x_coords <= 0) |
                      np.isnan(y_coords) | np.isinf(y_coords) | (y_coords <= 0))
        if valid_mask.sum() > 0:
            return x_coords, y_coords, valid_mask
        else:
            raise ValueError(f"[{sample_name}] adata.obsm['spatial'] no valid coordinates")
    
    raise KeyError(f"[{sample_name}] no space coordinates were found")

def get_spot_coordinates(adata, sample_name):
    x_coords, y_coords, _ = get_spatial_coords(adata, sample_name)
    return np.vstack([x_coords, y_coords]).T
# ------------------------Verification of spot alignment between genes and images----------------------------
def verify_spot_alignment(adata_gene, adata_img, sample_name, save_dir):
    verification_results = {
        "sample": sample_name,
        "gene_spot_count": adata_gene.n_obs,
        "image_spot_count": adata_img.n_obs,
        "length_match": False,
        "barcode_set_match": False,
        "barcode_order_match": False,
        "spatial_coordinate_match": False,
        "alignment_status": "FAIL"
    }

    if adata_gene.n_obs != adata_img.n_obs:
        verification_results["alignment_status"] = "FAIL: Length mismatch"
        report_str = f"Inconsistent lengths: ST data {adata_gene.n_obs} Image data {adata_img.n_obs} \n"
        print(report_str)
        raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")
    else:
        verification_results["length_match"] = True
       
        gene_barcodes = set(adata_gene.obs_names.astype(str))
        img_barcodes = set(adata_img.obs_names.astype(str))
        
        if gene_barcodes == img_barcodes:
            verification_results["barcode_set_match"] = True

            common_barcodes = list(gene_barcodes) 
            img_barcode_to_idx = {bc: i for i, bc in enumerate(adata_img.obs_names.astype(str))}
            new_order = [img_barcode_to_idx[bc] for bc in adata_gene.obs_names.astype(str)]
            adata_img_reordered = adata_img[new_order].copy()

            if all(adata_gene.obs_names.astype(str) == adata_img_reordered.obs_names.astype(str)):
                verification_results["barcode_order_match"] = True
            else:
                verification_results["alignment_status"] = "FAIL: Barcode order mismatch after reordering"
                raise RuntimeError(f"\n{sample_name} Spot alignment verification failed:{verification_results['alignment_status']}")
            

            gene_coords = get_spot_coordinates(adata_gene, sample_name)
            img_coords = get_spot_coordinates(adata_img_reordered, sample_name)
            spatial_diff = np.abs(gene_coords - img_coords).max()
            if spatial_diff < 1e-3:
                verification_results["spatial_coordinate_match"] = True
            else:
                verification_results["alignment_status"] = "FAIL: Spatial coordinate mismatch"
                raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")

            return adata_img_reordered
            
        else:
            only_gene = gene_barcodes - img_barcodes
            only_img = img_barcodes - gene_barcodes
            verification_results["alignment_status"] = "FAIL: Barcode set mismatch"
            report_str = f"barcode set inconsistency\n"
            if only_gene:
                report_str += f"Only in the barcode section of the genetic data:{list(only_gene)[:5]}...\n"
            if only_img:
                report_str += f"Only the barcodes in the image data:{list(only_img)[:5]}...\n"
            print(report_str)
            raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")

    verification_df = pd.DataFrame([verification_results])
    report_path = os.path.join(save_dir, f"{sample_name}_spot_alignment_verification.csv")
    verification_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return adata_img  
# ---------------------------- Resolution acquisition function ----------------------------
def get_sample_pixel_resolution(sample_name, sample_dir):
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    if not os.path.exists(metadata_path):
        print(f"[{sample_name}] metadata file not found:{metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
        if 'pixel_size_um_estimated' in metadata:
            pixel_resolution = float(metadata['pixel_size_um_estimated'])
            print(f"[{sample_name}] resolution:{pixel_resolution:.6f} μm/pixel")
            return pixel_resolution
        else:
            print(f"[{sample_name}] metadata has no 'pixel_size_um_estimated' ")
            return None
    except Exception as e:
        print(f"[{sample_name}] Failed to read metadata:{str(e)[:50]}")
        return None

def get_real_spot_pixel_radius(sample_name, sample_dir):

    pixel_resolution = get_sample_pixel_resolution(sample_name, sample_dir)
    if pixel_resolution is None:
        print(f"[{sample_name}] Unable to obtain resolution")
        return None
    
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'spot_diameter' in metadata:
                spot_physical_diam = float(metadata['spot_diameter'])
                print(f"[{sample_name}] Spot physical diameter:{spot_physical_diam:.2f} μm")
        except Exception as e:
            print(f"[{sample_name}] spot has no physical diameter")
    
    spot_pixel_diam = spot_physical_diam / pixel_resolution
    spot_pixel_radius = spot_pixel_diam / 2.0  
    print(f"[{sample_name}] actual Spot pixel radius:{spot_pixel_radius:.2f} pixel")
    return spot_pixel_radius
# ---------------------------- Read the outline of the tissue ----------------------------
def load_polygon_for_sample(sample_dir, sample_name):
    geojson_path = os.path.join(sample_dir, "tissue_seg", f"{sample_name}_contours.geojson")
    if not os.path.exists(geojson_path):
        return None, geojson_path
    try:
        gdf = gpd.read_file(geojson_path)
        polys = [geom for geom in gdf.geometry if geom is not None]
        if len(polys) == 0:
            return None, geojson_path
        if len(polys) == 1:
            return polys[0], geojson_path
        return unary_union(polys), geojson_path  
    except Exception as e:
        try:
            with open(geojson_path, 'r', encoding='utf-8') as fh:
                gj = json.load(fh)
            feats = gj.get('features', None)
            if feats is None:
                geom = gj.get('geometry', gj)
                shp = shapely_shape(geom)
                return shp, geojson_path
            polys = []
            for f in feats:
                g = f.get('geometry', None)
                if g is None:
                    continue
                polys.append(shapely_shape(g))
            if len(polys) == 0:
                return None, geojson_path
            if len(polys) == 1:
                return polys[0], geojson_path
            return unary_union(polys), geojson_path  
        except Exception as e2:
            print(f"Unable to read geojson：{e} / {e2}")
            return None, geojson_path
# ---------------------------Read the cell segmentation data--------------------------
def load_cellvit_polygons(sample_dir, sample_name):
    fname = os.path.join(sample_dir, CELLVIT_DIRNAME, CELLVIT_FILENAME_TEMPLATE.format(sample=sample_name))
    if not os.path.exists(fname):
        return None, None, fname
    try:
        with open(fname, 'r', encoding='utf-8') as fh:
            gj = json.load(fh)
    except Exception as e:
        warnings.warn(f"Unable to read cellvit geojson: {fname} - {e}")
        return None, None, fname

    if isinstance(gj, list):
        features = gj
    elif isinstance(gj, dict):
        features = gj.get('features', None)
        if features is None:
            geom = gj.get('geometry', gj)
            features = [{'geometry': geom, 'properties': gj.get('properties', {})}]
    else:
        warnings.warn(f"Unknown JSON structure: {fname}")
        return None, None, fname

    cell_polygons = []
    cell_types = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get('geometry', None)
        props = feat.get('properties', {})
        cls = None
        if 'classification' in props:
            cls = props['classification'].get('name', None) if isinstance(props['classification'], dict) else props['classification']
        if cls is None:
            cls = props.get('label', props.get('name', None))
        if geom is None:
            continue
        try:
            shp = shapely_shape(geom)
            if shp is None:
                continue
            if isinstance(shp, MultiPolygon):
                for p in shp.geoms:
                    cell_polygons.append(p)
                    cell_types.append(cls if cls is not None else "Unknown")
            else:
                cell_polygons.append(shp)
                cell_types.append(cls if cls is not None else "Unknown")
        except Exception:
            continue
    n_features = len(features)
    n_polygons = len(cell_polygons)    
    print(f"  geojson features数: {n_features}")
    print(f"  atomic polygon 数: {n_polygons}")
    return cell_polygons, cell_types, fname

# -------------------------- Calculation of tumor cell proportion---------------------------
def compute_spot_cell_counts_from_seg_fast_with_composition(adata, cell_polygons, cell_types, sample_name, sample_dir):
    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample_name)
    n_total_spots = len(x_coords)    
    if not valid_mask.any():
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    spot_r = get_real_spot_pixel_radius(sample_name, sample_dir)
    if spot_r is None:
        print(f"[{sample_name}] Unable to obtain the spot radius")
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    spot_geoms = []
    spot_original_idx = []
    spot_barcodes = []
    for idx in range(n_total_spots):
        if not valid_mask[idx]:
            continue
        x, y = x_coords[idx], y_coords[idx]
        spot_geoms.append(Point(x, y).buffer(spot_r))
        spot_original_idx.append(idx)
        spot_barcodes.append(adata.obs_names[idx])
    
    if len(spot_original_idx) == 0:
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)
    
    spots_gdf = gpd.GeoDataFrame({
        'spot_original_idx': spot_original_idx,
        'spot_barcode': spot_barcodes
    }, geometry=spot_geoms, crs="EPSG:4326").rename_geometry('geometry_spot')

    cell_ids = np.arange(len(cell_polygons))
    valid_cell_data = []
    rejected_records = []
    for cid, geom, ctype in zip(cell_ids, cell_polygons, cell_types):
        reason = None
        if geom is None:
            reason = "no_geom"
        elif ctype is None:
            reason = "no_type"
        else:
            if not isinstance(geom, (Polygon, MultiPolygon)):
                reason = "not_polygon"
            else:
                try:
                    if not geom.is_valid or geom.area == 0:
                        reason = "invalid_geometry"
                except Exception:
                    reason = "geom_check_error"
        if reason is not None:
            try:
                cx, cy = (geom.centroid.x, geom.centroid.y) if geom is not None else (np.nan, np.nan)
            except Exception:
                cx, cy = (np.nan, np.nan)
            rejected_records.append({"orig_cell_id": int(cid), "reason": reason, 
                                    "centroid_x": cx, "centroid_y": cy, 
                                    "cell_type": str(ctype) if ctype is not None else None})
            continue
        valid_cell_data.append({
            'cell_id': int(cid), 
            'cell_type': str(ctype).strip().lower(), 
            'geometry_cell': geom,
            'cell_centroid_x': geom.centroid.x,
            'cell_centroid_y': geom.centroid.y
        })

    cells_geoms = [d['geometry_cell'] for d in valid_cell_data]
    cells_gdf = gpd.GeoDataFrame(valid_cell_data, geometry=cells_geoms, 
                                crs="EPSG:4326").rename_geometry('temp_cell_geom')
    print(f"[{sample_name}] effective cell count:{len(cells_gdf)}")

    sample_save_dir = os.path.join(SAVE_ROOT, sample_name)
    os.makedirs(sample_save_dir, exist_ok=True)
    if len(rejected_records) > 0:
        df_rej = pd.DataFrame(rejected_records)
        rej_path = os.path.join(sample_save_dir, f"{sample_name}_filtered_cells.csv")
        df_rej.to_csv(rej_path, index=False, encoding="utf-8-sig")

    # Spatial connection
    try:
        join_res = gpd.sjoin(spots_gdf, cells_gdf, how='inner', predicate='intersects', 
                            lsuffix='_spot', rsuffix='_cell')
        if 'temp_cell_geom' in join_res.columns:
            join_res = join_res.drop(columns=['temp_cell_geom'])
    except Exception as e:
        raise RuntimeError(f"[{sample_name}] Space connection failed:{e}")

    # Calculate the overlapping area
    def calc_overlap_area(row):
        try:
            spot_geom = row['geometry_spot']
            cell_geom = row['geometry_cell']
            if not (isinstance(spot_geom, (Polygon, MultiPolygon)) and spot_geom.is_valid):
                return 0.0
            if not (isinstance(cell_geom, (Polygon, MultiPolygon)) and cell_geom.is_valid):
                return 0.0
            inter = spot_geom.intersection(cell_geom)
            return max(inter.area, 0.0) if inter and inter.is_valid else 0.0
        except Exception:
            return 0.0

    join_res['overlap_area'] = join_res.apply(calc_overlap_area, axis=1)
    join_res = join_res[join_res['overlap_area'] > 1e-6].copy()
    if len(join_res) == 0:
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    # 1cell->1spot（Select the spot with the largest overlapping area）
    join_res_sorted = join_res.sort_values(by=['cell_id', 'overlap_area'], ascending=[True, False])
    join_res_unique = join_res_sorted.drop_duplicates(subset='cell_id', keep='first')
    print(f"[{sample_name}] After removing duplicates:{len(join_res_unique)}")

    # Construct a DataFrame based on cell composition
    cell_composition_df = pd.DataFrame({
        'spot_index': join_res_unique['spot_original_idx'].values,
        'spot_barcode': join_res_unique['spot_barcode'].values,
        'cell_id': join_res_unique['cell_id'].values,
        'cell_type': join_res_unique['cell_type'].values,
        'overlap_area': join_res_unique['overlap_area'].values,
        'cell_centroid_x': join_res_unique['cell_centroid_x'].values,
        'cell_centroid_y': join_res_unique['cell_centroid_y'].values
    })

    spot_coords = {row['spot_original_idx']: (x_coords[row['spot_original_idx']], y_coords[row['spot_original_idx']]) 
                   for _, row in spots_gdf.iterrows()}
    cell_composition_df['spot_center_x'] = cell_composition_df['spot_index'].map(lambda x: spot_coords[x][0])
    cell_composition_df['spot_center_y'] = cell_composition_df['spot_index'].map(lambda x: spot_coords[x][1])

    total_count = np.zeros(n_total_spots, int)
    tumor_count = np.zeros(n_total_spots, int)

    spot_cell_stats = join_res_unique.groupby('spot_original_idx')['cell_id'].nunique()
    for spot_idx, count in spot_cell_stats.items():
        total_count[spot_idx] = int(count)
    
    # Statistics of Tumor Cells
    tumor_mask = join_res_unique['cell_type'].str.startswith('neoplastic', na=False)
    if tumor_mask.any():
        tumor_cells_df = join_res_unique[tumor_mask]
        spot_tumor_stats = tumor_cells_df.groupby('spot_original_idx')['cell_id'].nunique()
        for spot_idx, count in spot_tumor_stats.items():
            tumor_count[spot_idx] = int(count)
    else:
        print(f"[{sample_name}] No tumor cells were detected")

    # Calculate the proportion of tumors
    tumor_ratio = np.zeros(n_total_spots, float)
    for i in range(n_total_spots):
        if total_count[i] > 0:
            tumor_ratio[i] = tumor_count[i] / total_count[i]
        else:
            tumor_ratio[i] = 0.0
    
    return tumor_count, total_count, tumor_ratio, cell_composition_df
# ---------------------------Similarity indicator calculation function ----------------------------
def compute_correlation_metrics(gene_ptime, img_ptime):
    """Correlation coefficient type indicators"""
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 3:
        print(f"Insufficient valid data points")
        return {k: np.nan for k in ['pearson_r', 'pearson_p', 'kendall_tau', 'kendall_p', 
                                   'spearman_r', 'spearman_p', 'distance_corr']}
    
    # Pearson
    pearson_r, pearson_p = pearsonr(x, y)
    # Kendall tau
    kendall_tau, kendall_p = kendalltau(x, y)
    # Spearman
    spearman_r, spearman_p = spearmanr(x, y)
    # Distance Correlation
    def distance_correlation(a, b):
        def _centered_distance_matrix(x):
            n = len(x)
            dist = cdist(x.reshape(-1, 1), x.reshape(-1, 1), 'euclidean')
            row_mean = dist.mean(axis=1, keepdims=True)
            col_mean = dist.mean(axis=0, keepdims=True)
            total_mean = dist.mean()
            return dist - row_mean - col_mean + total_mean
        dx = _centered_distance_matrix(x)
        dy = _centered_distance_matrix(y)
        cov_dxy = (dx * dy).sum() / (n ** 2)
        cov_dx = (dx ** 2).sum() / (n ** 2)
        cov_dy = (dy ** 2).sum() / (n ** 2)
        return np.sqrt(cov_dxy / np.sqrt(cov_dx * cov_dy)) if cov_dx > 0 and cov_dy > 0 else 0.0
    distance_corr = distance_correlation(x, y)

    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'kendall_tau': kendall_tau, 'kendall_p': kendall_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'distance_corr': distance_corr
    }

def compute_ranking_metrics(gene_ptime, img_ptime):
    """Sorting consistency index"""
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 2:
        print(f"Insufficient valid data points")
        return {k: np.nan for k in ['kendall_tau_b', 'kendall_tau_b_p', 'tau_distance', 
                                   'concordant_pairs', 'discordant_pairs']}

    rank_x = stats.rankdata(x, method='average')
    rank_y = stats.rankdata(y, method='average')
    
    # Kendall tau-b
    def kendall_tau_b(r1, r2):
        n = len(r1)
        concordant = 0
        discordant = 0
        ties_x = 0
        ties_y = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = r1[i] - r1[j]
                dy = r2[i] - r2[j]
                if dx * dy > 0:
                    concordant += 1
                elif dx * dy < 0:
                    discordant += 1
                else:
                    if dx == 0:
                        ties_x += 1
                    if dy == 0:
                        ties_y += 1
        numerator = concordant - discordant
        denominator = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
        tau_b = numerator / denominator if denominator != 0 else 0.0
        se = np.sqrt((n*(n-1)*(2*n+5) - ties_x*(2*n+5) - ties_y*(2*n+5) + 4*ties_x*ties_y/n) / (18*(n-1)*n**2))
        z = tau_b / se if se != 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
        return tau_b, p_val, concordant, discordant
    
    kendall_tau_b_val, kendall_tau_b_p, concordant, discordant = kendall_tau_b(rank_x, rank_y)
    # Tau distance
    tau_distance = 0.5 * (1 - (concordant - discordant) / comb(n, 2)) if n >=2 else np.nan
    
    return {
        'kendall_tau_b': kendall_tau_b_val, 'kendall_tau_b_p': kendall_tau_b_p,
        'tau_distance': tau_distance,
        'concordant_pairs': concordant,
        'discordant_pairs': discordant
    }

def compute_distribution_metrics(gene_ptime, img_ptime):
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    if len(x) < 2 or len(y) < 2:
        print(f"Insufficient valid data points")
        return {'wasserstein_distance': np.nan}
    
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    wasserstein_dist = wasserstein_distance(x_norm, y_norm)
    return {'wasserstein_distance': wasserstein_dist}

def compute_cosine_similarity(gene_ptime, img_ptime):
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 2:
        print(f"Insufficient valid data points")
        return np.nan
    
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    dot_product = np.dot(x_norm, y_norm)
    x_norm_l2 = np.linalg.norm(x_norm)
    y_norm_l2 = np.linalg.norm(y_norm)
    
    return dot_product / (x_norm_l2 * y_norm_l2) if x_norm_l2 > 0 and y_norm_l2 > 0 else 0.0

# --------------------------- Visualization function for the proportion of spot tumors -------------------------
def plot_tumor_ratio_spatial(adata_before, adata_after, sample_name, save_dir, sample_dir=None):

    if 'tumor_ratio' not in adata_before.obs.columns:
        print(f"{sample_name} no tumor_ratio")
        return

    if 'cell_count' not in adata_before.obs.columns:
        print(f"{sample_name} no cell_count")
        return

    if sample_dir is None:
        sample_dir = os.path.join(ROOT_DIR, sample_name)

    poly, poly_path = load_polygon_for_sample(sample_dir, sample_name)
    has_tissue_contour = poly is not None

    kept_spots = set(adata_after.obs_names)
    mask_kept = adata_before.obs_names.isin(kept_spots)

    x_coords, y_coords, _ = get_spatial_coords(adata_before, sample_name)
    tumor_ratio = adata_before.obs['tumor_ratio'].astype(float).values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    if mask_kept.any():
        sc = ax.scatter(
            x_coords[mask_kept],
            y_coords[mask_kept],
            c=tumor_ratio[mask_kept],
            s=25, cmap='Reds',
            vmin=0, vmax=1,
            edgecolors='k', linewidths=0.3,
            alpha=0.9, zorder=3
        )
    else:
        sc = None
        print(f"{sample_name}: unreserved spot")

    if has_tissue_contour:
        def plot_polygon(ax_obj, geom, color='black', linewidth=2.0, alpha=0.7):
            if isinstance(geom, Polygon):
                x_poly, y_poly = geom.exterior.xy
                ax_obj.plot(x_poly, y_poly, color=color, linewidth=linewidth,
                            alpha=alpha, zorder=5)
                for interior in geom.interiors:
                    xi, yi = interior.xy
                    ax_obj.plot(xi, yi, color=color, linewidth=linewidth-0.5,
                                alpha=alpha, linestyle='--', zorder=5)
            elif isinstance(geom, MultiPolygon):
                for sub_geom in geom.geoms:
                    plot_polygon(ax_obj, sub_geom, color, linewidth, alpha)

        plot_polygon(ax, poly)
        print(f"Organizational outline has been added: {poly_path}")

    ax.invert_yaxis()
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.tick_params(labelsize=11)
    ax.legend(loc='upper right', fontsize=28)

    if sc is not None:
        cbar_ax = fig.add_axes([0.82, 0.1, 0.03, 0.8])
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Tumor Ratio", fontsize=28)
        cbar.ax.tick_params(labelsize=28, width=1.5, length=4)

    save_path = os.path.join(save_dir, f"{sample_name}_tumor_ratio_spatial.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"The tumor proportion space graph has been saved:{save_path}")
# ---------------------------- Pseudotime visualization function -------------------------
def plot_gene_pseudotime_distribution(gene_ptime, sample_name, save_dir):
    gene_ptime_clean = gene_ptime[np.isfinite(gene_ptime)]
    
    print(f"Number of effective values of gene pseudotime:{len(gene_ptime_clean)}/{len(gene_ptime)}")
    
    if len(gene_ptime_clean) == 0:
        print("There is no valid pseudotime data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(gene_ptime_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Pseudotime', fontsize=28)
    ax.set_ylabel('Frequency', fontsize=28)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_gene_pseudotime_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The pseudotime distribution map of genes has been saved:{save_path}")

def plot_image_pseudotime_distribution(img_ptime, sample_name, save_dir):
    img_ptime_clean = img_ptime[np.isfinite(img_ptime)]
    
    print(f"Number of effective values of image pseudotime:{len(img_ptime_clean)}/{len(img_ptime)}")
    
    if len(img_ptime_clean) == 0:
        print("There is no valid pseudotime data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.hist(img_ptime_clean, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Pseudotime', fontsize=28)
    ax.set_ylabel('Frequency', fontsize=28)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_image_pseudotime_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The pseudotime distribution map of image has been saved: {save_path}")
# ---------------------------- UMAP visualization function ----------------------------
def plot_gene_umap_pseudotime(gene_adata, gene_ptime, img_ptime, sample_name, save_dir):

    try:
        sc.pp.neighbors(gene_adata, use_rep='X_pca')
        sc.tl.umap(gene_adata, min_dist=0.5, spread=1.0)

        umap_coords = gene_adata.obsm['X_umap']

        gene_adata.obsm['X_umap_gene'] = umap_coords.copy()
        COMMON_CMAP = 'viridis'

        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_axes([0.12, 0.12, 0.68, 0.82])
        
        scatter_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                 c=gene_ptime, 
                                 cmap=COMMON_CMAP,
                                 s=60,
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=1.2)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)        
        ax.set_xlabel('UMAP1', fontsize=28, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=28, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        ax.grid(True, alpha=0.2, linewidth=1.2)
        
        if 'iroot' in gene_adata.uns:
            root_idx = gene_adata.uns['iroot']
            ax.scatter(umap_coords[root_idx, 0], umap_coords[root_idx, 1], 
                      color='red', s=450, marker='o', edgecolor='white', 
                      linewidths=3, label='Root', zorder=10)
            ax.legend(fontsize=28, prop={'weight': 'bold'},
                     frameon=True, edgecolor='black', framealpha=0.9,
                     loc='upper right')
  
        cbar_ax = fig.add_axes([0.82, 0.12, 0.03, 0.82])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Gene Pseudotime', fontsize=28, fontweight='bold')
        cbar.ax.tick_params(labelsize=28, width=2.0, length=6)
        cbar.outline.set_linewidth(2.0)
        
        save_path1 = os.path.join(save_dir, f"{sample_name}_gene_umap_gene_pseudotime.png")
        plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Gene characteristics UMAP:{save_path1}")

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_axes([0.12, 0.12, 0.68, 0.82])
        
        scatter_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                 c=img_ptime, 
                                 cmap=COMMON_CMAP,
                                 s=60,
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=1.2)
        
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel('UMAP1', fontsize=28, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=28, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        ax.grid(True, alpha=0.2, linewidth=1.2)
        
        if 'iroot' in gene_adata.uns:
            root_idx = gene_adata.uns['iroot']
            ax.scatter(umap_coords[root_idx, 0], umap_coords[root_idx, 1], 
                      color='red', s=450, marker='o', edgecolor='white', 
                      linewidths=3, label='Root', zorder=10)
            ax.legend(fontsize=28, prop={'weight': 'bold'},
                     frameon=True, edgecolor='black', framealpha=0.9,
                     loc='upper right')
        
        cbar_ax = fig.add_axes([0.82, 0.12, 0.03, 0.82])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Image Pseudotime', fontsize=28, fontweight='bold')
        cbar.ax.tick_params(labelsize=28, width=2.0, length=6)
        cbar.outline.set_linewidth(2.0)
        
        save_path2 = os.path.join(save_dir, f"{sample_name}_gene_umap_image_pseudotime.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Gene characteristics UMAP:{save_path2}")
        
        return gene_adata
        
    except Exception as e:
        return None

def plot_image_umap_pseudotime(img_adata, gene_ptime, img_ptime, sample_name, save_dir):
    try:
        sc.pp.neighbors(img_adata, use_rep='X_pca')
        sc.tl.umap(img_adata, min_dist=0.5, spread=1.0)
        
        umap_coords = img_adata.obsm['X_umap']

        img_adata.obsm['X_umap_image'] = umap_coords.copy()
        
        COMMON_CMAP = 'viridis'

        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_axes([0.12, 0.12, 0.68, 0.82])
        
        scatter_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                 c=gene_ptime, 
                                 cmap=COMMON_CMAP,
                                 s=60,
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=1.2)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel('UMAP1', fontsize=28, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=28, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        ax.grid(True, alpha=0.2, linewidth=1.2)
        
        if 'iroot' in img_adata.uns:
            root_idx = img_adata.uns['iroot']
            ax.scatter(umap_coords[root_idx, 0], umap_coords[root_idx, 1], 
                      color='red', s=450, marker='o', edgecolor='white', 
                      linewidths=3, label='Root', zorder=10)
            ax.legend(fontsize=28, prop={'weight': 'bold'},
                     frameon=True, edgecolor='black', framealpha=0.9,
                     loc='upper right')
        
        cbar_ax = fig.add_axes([0.82, 0.12, 0.03, 0.82])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Gene Pseudotime', fontsize=28, fontweight='bold')
        cbar.ax.tick_params(labelsize=28, width=2.0, length=6)
        cbar.outline.set_linewidth(2.0)
        
        save_path1 = os.path.join(save_dir, f"{sample_name}_image_umap_gene_pseudotime.png")
        plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Image feature UMAP:{save_path1}")
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_axes([0.12, 0.12, 0.68, 0.82])
        
        scatter_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                 c=img_ptime, 
                                 cmap=COMMON_CMAP,
                                 s=60,
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=1.2)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel('UMAP1', fontsize=28, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=28, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        ax.grid(True, alpha=0.2, linewidth=1.2)
        
        if 'iroot' in img_adata.uns:
            root_idx = img_adata.uns['iroot']
            ax.scatter(umap_coords[root_idx, 0], umap_coords[root_idx, 1], 
                      color='red', s=450, marker='o', edgecolor='white', 
                      linewidths=3, label='Root', zorder=10)
            ax.legend(fontsize=28, prop={'weight': 'bold'},
                     frameon=True, edgecolor='black', framealpha=0.9,
                     loc='upper right')
        
        cbar_ax = fig.add_axes([0.82, 0.12, 0.03, 0.82])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Image Pseudotime', fontsize=28, fontweight='bold')
        cbar.ax.tick_params(labelsize=28, width=2.0, length=6)
        cbar.outline.set_linewidth(2.0)
        
        save_path2 = os.path.join(save_dir, f"{sample_name}_image_umap_image_pseudotime.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Image feature UMAP: {save_path2}")
        
        return img_adata
        
    except Exception as e:
        return None
# ---------------------------- Visual function -------------------------------
def plot_and_save(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
def save_spatial_with_root(adata_for_plot, color_key, root_coord, save_path, sample_name):
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    sc.pl.spatial(
        adata_for_plot,
        color=[color_key],
        img_key="downscaled_fullres",  
        size=1.3,
        ncols=1,
        show=False,
        cmap="viridis",
        title='', 
        ax=ax
    )

    children = list(ax.get_children())
    cbar_to_remove = None
    for child in children:
        if hasattr(child, 'colorbar') and child.colorbar is not None:
            cbar_to_remove = child.colorbar
            break
        elif isinstance(child, plt.Axes) and child.get_label() == '<colorbar>':
            cbar_to_remove = child
            break
    if cbar_to_remove is not None:
        cbar_to_remove.remove()
           

    x_coords, y_coords, _ = get_spatial_coords(adata_for_plot, sample_name)
        
    if 'iroot' in adata_for_plot.uns:
        root_spot_name = adata_for_plot.obs_names[adata_for_plot.uns['iroot']]
        root_index = list(adata_for_plot.obs_names).index(root_spot_name)
    else:
        dists = np.sqrt((x_coords - root_coord[0])**2 + (y_coords - root_coord[1])**2)
        root_index = np.argmin(dists)
        
    sf = adata_for_plot.uns['spatial']['ST']['scalefactors']
    scale = sf['tissue_downscaled_fullres_scalef']
    spot_diam_fullres = sf['spot_diameter_fullres']
    spot_diam_plot = spot_diam_fullres * scale
    marker_size = (spot_diam_plot * 1.8) ** 2
            
    x_full, y_full = x_coords[root_index], y_coords[root_index]
    x_plot = x_full * scale
    y_plot = y_full * scale
   
    ax.scatter([x_plot], [y_plot],
            color='white',
            s=marker_size * 1.5,  
            marker='o',
            edgecolor='white',
            linewidths=1.5,
            zorder=9,  
            label='_nolegend_')  
    ax.scatter([x_plot], [y_plot],
            color='red',
            s=marker_size,
            marker='o',
            edgecolor='white',
            linewidths=1.5,
            zorder=10,
            label='Root')

    ax.legend(loc='upper right', frameon=True, framealpha=0.8, fontsize=28)
    ax_pos = ax.get_position()
    new_ax_width = 0.7  
    new_ax_left = 0.1   
    new_ax_bottom = 0.1  
    new_ax_height = 0.8  
    ax.set_position([new_ax_left, new_ax_bottom, new_ax_width, new_ax_height])
      
    cbar_left = new_ax_left + new_ax_width + 0.02  
    cbar_width = 0.03 
    cbar_bottom = new_ax_bottom  
    cbar_height = new_ax_height  
        
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        
    vals = adata_for_plot.obs[color_key].astype(float).values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
     
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(color_key.replace('_', ' ').title(), fontsize=28)
    cbar.ax.tick_params(labelsize=28)
      
    ax.set_xlabel('X coordinate', fontsize=28)
    ax.set_ylabel('Y coordinate', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ----------------------------- Organizational filtering function ----------------------------
def filter_spots_by_tissue(adata, sample, sample_dir, save_dir):
    original_spots = adata.n_obs

    poly, poly_path = load_polygon_for_sample(sample_dir, sample)
    if poly is None:
        print(f"{poly_path} no organization outline file was found.")
        return adata, np.ones(adata.n_obs, dtype=bool), original_spots, original_spots

    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample)
    if not np.any(valid_mask):
        print("Without valid coordinates, organization filtering cannot be performed")
        return adata, valid_mask, original_spots, original_spots
    
    spot_r = get_real_spot_pixel_radius(sample, sample_dir)

    inside_mask = np.zeros(len(x_coords), dtype=bool)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if not valid_mask[i]:
            inside_mask[i] = False
            continue
            
        try:
            spot_circle = Point(x, y).buffer(spot_r)
            center_inside = poly.contains(Point(x, y))
            intersection = poly.intersection(spot_circle)
            area_inside = intersection.area / spot_circle.area if spot_circle.area > 0 else 0
            inside_mask[i] = center_inside or (area_inside > 0.7)
            
        except Exception as e:
            print(f"Error occurred while processing coordinates ({x}, {y}): {e} → Filtering")
            inside_mask[i] = False
    
    n_total = len(inside_mask)
    n_inside = int(inside_mask.sum())
    n_filtered = n_total - n_inside
    retention_rate = n_inside / n_total if n_total > 0 else 0
    print(f"tissue filter：Total spots{n_total} Reserved spots{n_inside} Filter spots{n_filtered}")
    

    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        def _plot_polygon(ax_obj, geom):
            if isinstance(geom, Polygon):
                x_poly, y_poly = geom.exterior.xy
                ax_obj.plot(x_poly, y_poly, color='red', linewidth=1.6)
                for interior in geom.interiors:
                    xi, yi = interior.xy
                    ax_obj.plot(xi, yi, color='red', linewidth=0.8, alpha=0.7)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    _plot_polygon(ax_obj, sub)
        _plot_polygon(ax, poly)

        ax.scatter(
            x_coords[inside_mask & valid_mask], 
            y_coords[inside_mask & valid_mask], 
            s=8, c="blue", label=f"Inside Tissue ({n_inside} spots)"
        )
        ax.scatter(
            x_coords[~inside_mask & valid_mask], 
            y_coords[~inside_mask & valid_mask], 
            s=6, c="gray", alpha=0.5, label=f"Outside Tissue ({n_filtered} spots)"
        )
        ax.scatter(
            x_coords[~valid_mask], 
            y_coords[~valid_mask], 
            s=4, c="black", alpha=0.3, label=f"Invalid Coords ({len(x_coords)-valid_mask.sum()} spots)"
        )
        
        ax.invert_yaxis()
        ax.legend(fontsize=12)
        plot_and_save(fig, os.path.join(save_dir, f"{sample}_tissue_filter_check.png"))

    except Exception as e_vis:
        print(f"Failed to generate the organizational filtering result graph:{e_vis}")

    filtered_adata = adata[inside_mask].copy()
    return filtered_adata, inside_mask, original_spots, filtered_adata.n_obs
# ----------------------------- Cell count filtering function ----------------------------
def filter_spots_by_cell_count_enhanced(adata, sample_name=""):
    if 'cell_count' not in adata.obs.columns:
        mask = np.ones(adata.n_obs, dtype=bool)
        return adata.copy(), mask, None, 0
    cell_counts = adata.obs['cell_count'].astype(int).values

    mask_keep = (cell_counts > 0)
    n_total = len(mask_keep)
    n_kept = int(mask_keep.sum())
    n_filtered = n_total - n_kept
    
    filtered_adata = adata[mask_keep].copy()
    final_threshold = 1  
    return filtered_adata, mask_keep, cell_counts, final_threshold

# ---------------------------The function for checking the validity of pseudotime----------------------------
def check_pseudotime_validity_strict(ptime_array, sample_name, ptime_type):
    nan_count = np.sum(np.isnan(ptime_array))
    inf_count = np.sum(np.isinf(ptime_array))
    total_count = len(ptime_array)
    
    print(f"[{sample_name}] {ptime_type}Pseudotime validity check:")
    if nan_count > 0 or inf_count > 0:
        print(f"Discovering invalid values:NaN {nan_count} INF {inf_count} ")
        return False, nan_count, inf_count
    
    print(f"All pseudotime values are valid")
    return True, nan_count, inf_count

# --------------------Root node selection function ----------------------------
def select_root_cluster_simple(cluster_tumor_stats, sample_name):

    if not cluster_tumor_stats:
        return None

    valid_stats = {}
    for cluster_id, stats in cluster_tumor_stats.items():
        if not np.isnan(stats['avg_tumor_ratio']):
            valid_stats[cluster_id] = stats
    
    if not valid_stats:
        return None
    
    print(f"\n[{sample_name}] Root node cluster selection:")
    print(f"Number of effective clusters: {len(valid_stats)}")
   

    min_tumor_ratio = min(stats['avg_tumor_ratio'] for stats in valid_stats.values())
    

    candidate_clusters = []
    for cluster_id, stats in valid_stats.items():
        if abs(stats['avg_tumor_ratio'] - min_tumor_ratio) < 1e-6:  
            candidate_clusters.append((cluster_id, stats))
    
    print(f"The number of candidate clusters with the smallest tumor proportion:{len(candidate_clusters)}")
    
    if len(candidate_clusters) == 1:
        selected_cluster = candidate_clusters[0][0]
        print(f"There is only one cluster with the smallest proportion of tumors: {selected_cluster}")
        return selected_cluster

    print(f"There are {len(candidate_clusters)} clusters with the smallest tumor proportions, and the one with the most spots is selected.")

    best_cluster = max(candidate_clusters, key=lambda x: x[1]['size'])
    selected_cluster = best_cluster[0]
    print(f"Select the cluster with the largest number of spots {selected_cluster} (spot={best_cluster[1]['size']})")
    
    return selected_cluster
# ========================= Read the sample list from the txt file ===================
def read_sample_list_from_txt(file_path):
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    samples.append(line)
        print(f"Reading samples: {len(samples)} ")
        return samples
    except Exception as e:
        print(f"Failed to read the sample list file:{e}")
        return []
# ==================== Read the sample list from the CSV file ====================
def read_sample_list_from_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Sample list file does not exist:{file_path}")
        return []
    
    try:
        df = pd.read_csv(file_path)
        
        if 'sample' not in df.columns:
            print(f"The CSV file is missing the 'sample' column: {df.columns.tolist()}")
            return []
        
        samples = df['sample'].dropna().astype(str).str.strip().tolist()
        samples = [s for s in samples if s and not s.startswith('#')]
        
        print(f"Reading samples: {len(samples)} ")
        return samples
        
    except Exception as e:
        print(f"Failed to read the sample list file: {e}")
        return []

# ---------------------------- Main ----------------------------
# all samples
all_samples = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

#Select file type: 'csv' or 'txt'
SAMPLE_LIST_TYPE = 'csv'  
SAMPLE_LIST_FILE = "select_samples.csv" 

if SAMPLE_LIST_TYPE == 'csv':
    samples = read_sample_list_from_csv(SAMPLE_LIST_FILE)
else:
    samples = read_sample_list_from_txt(SAMPLE_LIST_FILE)

if not samples:
    print(f"The sample list file was not found or it was empty. All samples were read from the directory...")
    all_samples = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    samples = all_samples
    SAMPLE_TO_TISSUE = {} 
    print(f"Read {len(samples)} samples from the directory")

print(f"The number of samples to be processed: {len(samples)}")

summary_columns = [
    "sample", "n_valid_spots", "n_spots_with_cells", "reason",
    "original_spots", "after_qc", "after_tissue_filter", "after_cell_filter", "after_all_filters",
    "retention_rate_qc", "retention_rate_tissue", "retention_rate_cell", "retention_rate_total",
    "final_cell_threshold",
    "leiden_clusters", "start_cluster_id", "start_cluster_tumor_ratio", "start_cluster_size",
    "pearson_r", "pearson_p", "kendall_tau", "kendall_p", "spearman_r", "spearman_p","distance_corr",
    "kendall_tau_b", "kendall_tau_b_p", "tau_distance", "concordant_pairs", "discordant_pairs",
    "wasserstein_distance",
    "cosine_similarity",
    "tumor_ratio_mean", "tumor_ratio_max", "tumor_ratio_median",
    "cell_count_mean", "cell_count_max", "cell_count_median", "zero_cell_spots", "zero_cell_ratio",
    "total_cells_matched", "unique_cell_types", "avg_cells_per_spot",
    "gene_ptime_mean", "gene_ptime_std", "gene_ptime_min", "gene_ptime_max",
    "img_ptime_mean", "img_ptime_std", "img_ptime_min", "img_ptime_max"
]
summary_rows = []

for sample in tqdm(samples, desc="Samples"):
    sample_dir = os.path.join(ROOT_DIR, sample)
    st_path = os.path.join(sample_dir, "st", f"{sample}.h5ad")
    feat_path = os.path.join(sample_dir, "ext_feature", f"{sample}_patches_fix.h5")

    save_dir = os.path.join(SAVE_ROOT, sample)
    os.makedirs(save_dir, exist_ok=True)

    metrics = {col: np.nan for col in summary_columns if col not in ["sample", "reason"]}
    metrics["sample"] = sample
    metrics["reason"] = "OK"
    metrics["n_valid_spots"] = 0
    metrics["n_spots_with_cells"] = 0
    metrics["original_spots"] = 0
    metrics["after_qc"] = 0
    metrics["after_tissue_filter"] = 0
    metrics["after_cell_filter"] = 0
    metrics["after_all_filters"] = 0
    metrics["retention_rate_qc"] = 0.0
    metrics["retention_rate_tissue"] = 0.0
    metrics["retention_rate_cell"] = 0.0
    metrics["retention_rate_total"] = 0.0
    metrics["final_cell_threshold"] = 0
    metrics["leiden_clusters"] = 0
    metrics["start_cluster_id"] = ""
    metrics["start_cluster_tumor_ratio"] = np.nan
    metrics["start_cluster_size"] = 0
    metrics["zero_cell_spots"] = 0
    metrics["zero_cell_ratio"] = 0.0
    metrics["total_cells_matched"] = 0
    metrics["unique_cell_types"] = 0
    metrics["avg_cells_per_spot"] = 0.0

    if not os.path.exists(st_path):
        metrics["reason"] = "Missing ST file"
        print(f"[跳过] {sample}: {metrics['reason']}")
        summary_rows.append([metrics[col] for col in summary_columns])
        continue

    try:
  
        adata = sc.read_h5ad(st_path)
        metrics["original_spots"] = adata.n_obs
  
        # ==================== basic QC =====================
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
        metrics["after_qc"] = adata.n_obs
        metrics["retention_rate_qc"] = metrics["after_qc"] / metrics["original_spots"] if metrics["original_spots"] > 0 else 0
        print(f"QC 后 spots: {adata.n_obs} (保留率: {metrics['retention_rate_qc']*100:.1f}%)")

        # ===================tissue contour filtering=====================
        poly, poly_path = load_polygon_for_sample(sample_dir, sample)
        if poly is not None:
            adata, inside_mask, before_tissue, after_tissue = filter_spots_by_tissue(adata, sample, sample_dir, save_dir)
            metrics["after_tissue_filter"] = after_tissue
            metrics["retention_rate_tissue"] = after_tissue / metrics["after_qc"] if metrics["after_qc"] > 0 else 0
        else:
            print(f"No organization outline file was found: {poly_path}")
            metrics["reason"] = "Missing tissue file"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        
        # ==================== Cell count filtration and cell composition preservation =====================
        cell_polygons, cell_types, _ = load_cellvit_polygons(sample_dir, sample)
        seg_available = (cell_polygons is not None and len(cell_polygons) > 0)
        cell_composition_df = None
        
        if not seg_available:
            metrics["reason"] = f"Missing cell segmentation data"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        
        # Calculate the proportion and number of tumors
        tumor_count, total_count, tumor_ratio, cell_composition_df = compute_spot_cell_counts_from_seg_fast_with_composition(
            adata, cell_polygons, cell_types, sample, sample_dir
        )
        
        # Add the cell statistics information to adata
        adata.obs['tumor_ratio'] = tumor_ratio
        adata.obs['tumor_count'] = tumor_count
        adata.obs['cell_count'] = total_count
        
        # Record the number of spots before the filtration of cells
        before_cell_filter = adata.n_obs
        adata_before_cell_filter = adata.copy()  

        # Cell count filtration: 0 cells in the spot
        adata, cell_count_mask, cell_counts, final_threshold = filter_spots_by_cell_count_enhanced(
            adata, sample
        )
 
        metrics["final_cell_threshold"] = final_threshold
        
        # Record the number of filtered spots after cell counting
        after_cell_filter = adata.n_obs
        metrics["after_cell_filter"] = after_cell_filter
        metrics["retention_rate_cell"] = after_cell_filter / before_cell_filter if before_cell_filter > 0 else 0
        

        metrics["n_spots_with_cells"] = adata.n_obs
        metrics["zero_cell_spots"] = np.sum(cell_counts == 0) if cell_counts is not None else 0
        metrics["zero_cell_ratio"] = metrics["zero_cell_spots"] / len(cell_counts) if cell_counts is not None else 0.0
        metrics["cell_count_mean"] = np.mean(cell_counts) if cell_counts is not None else np.nan
        metrics["cell_count_max"] = np.max(cell_counts) if cell_counts is not None else np.nan
        metrics["cell_count_median"] = np.median(cell_counts) if cell_counts is not None else np.nan
        
        # Update Cell Composition Statistics
        if cell_composition_df is not None and not cell_composition_df.empty:
            filtered_spot_barcodes = adata.obs_names
            cell_composition_df_filtered = cell_composition_df[
                cell_composition_df['spot_barcode'].isin(filtered_spot_barcodes)
            ]
            
            metrics["total_cells_matched"] = len(cell_composition_df_filtered)
            metrics["unique_cell_types"] = cell_composition_df_filtered['cell_type'].nunique()
            metrics["avg_cells_per_spot"] = cell_composition_df_filtered.groupby('spot_barcode')['cell_id'].nunique().mean()
            
            cell_composition_path = os.path.join(save_dir, f"{sample}_cell_composition.csv")
            cell_composition_df_filtered.to_csv(cell_composition_path, index=False, encoding="utf-8-sig")

            spot_cell_type_summary = cell_composition_df_filtered.groupby(['spot_barcode', 'cell_type']).size().unstack(fill_value=0)
            spot_cell_type_summary['total_cells'] = spot_cell_type_summary.sum(axis=1)
            spot_cell_type_summary_path = os.path.join(save_dir, f"{sample}_spot_cell_type_summary.csv")
            spot_cell_type_summary.to_csv(spot_cell_type_summary_path, encoding="utf-8-sig")
        
        print(f"After cell count filtering, {adata.n_obs} spots remain")
        
        # ====================== Filter the samples based on the number of spots ===================== 
        n_spots = adata.n_obs
        if n_spots < MIN_SPOTS:
            metrics["reason"] = f"Spot count out of range ({n_spots})"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue

        # ====================The image features of the spot after matching filtering=====================      
        adata.layers['counts'] = adata.X.copy()
        adata = st.convert_scanpy(adata, use_quality='downscaled_fullres').copy()
        if 'spatial' not in adata.obsm:
            x_coords, y_coords, _ = get_spatial_coords(adata, sample)
            adata.obsm['spatial'] = np.vstack([x_coords, y_coords]).T

        if not os.path.exists(feat_path):
            metrics["reason"] = "Missing image feature file"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        
        with h5py.File(feat_path, 'r') as f:
            feats = f[FEATURE_KEY][()]    
            barcodes_img = np.array(f[BARCODE_KEY][()]).squeeze().astype(str)
        feat_map = {bc: feats[i] for i, bc in enumerate(barcodes_img)}
        
        adata_bcs = adata.obs_names.astype(str)
        keep_idx = [i for i, bc in enumerate(adata_bcs) if bc in feat_map]
        if len(keep_idx) == 0:
            metrics["reason"] = "No matched barcodes between gene and image data"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        
        adata_gene = adata[keep_idx, :].copy()
        img_feats = np.array([feat_map[bc] for bc in adata_gene.obs_names.astype(str)], dtype=np.float32)
        print(f"After image feature matching spots: {adata_gene.n_obs}")

        adata_img = sc.AnnData(img_feats)
        adata_img.obs = adata_gene.obs.copy()
        adata_img.obsm = adata_gene.obsm.copy()
        adata_img.uns = adata_gene.uns.copy()
        sc.pp.scale(adata_img)
        print(f"Image feature dimension: {adata_img.shape}")

        verify_spot_alignment(adata_gene, adata_img, sample, save_dir)
        

        metrics["after_all_filters"] = adata_gene.n_obs
        metrics["retention_rate_total"] = adata_gene.n_obs / metrics["original_spots"] if metrics["original_spots"] > 0 else 0

        # Highly variable gene extraction
        sc.pp.normalize_total(adata_gene)
        sc.pp.log1p(adata_gene)
        sc.pp.highly_variable_genes(adata_gene, n_top_genes=HVG_N)
        n_hvg = int(adata_gene.var['highly_variable'].sum())
        if n_hvg < N_PCS:
            metrics["reason"] = f"HVG count too low ({n_hvg} < {N_PCS})"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        adata_hvg = adata_gene[:, adata_gene.var['highly_variable']].copy()
        adata_gene=adata_hvg
        sc.pp.scale(adata_gene)
        print(f"Gene feature dimension: {adata_gene.shape}")

        # =====================Calculation of pseudo-time values of genes===========================   
        # 1.Dimensionality reduction and clustering
        sc.tl.pca(adata_gene, n_comps=N_PCS, random_state=0)
        sc.pp.neighbors(adata_gene, n_pcs=N_PCS, random_state=0)
        if DIFFMAP:
            sc.tl.diffmap(adata_gene)
            sc.pp.neighbors(adata_gene,  use_rep='X_diffmap', random_state=0)
        sc.tl.leiden(adata_gene, resolution=RESOLUTION, random_state=0)

        leiden_clusters = adata_gene.obs['leiden'].unique()
        metrics["leiden_clusters"] = len(leiden_clusters)
        print(f"Leiden cluster: {metrics['leiden_clusters']}")

        # 2. Root node cluster selection
        if seg_available:
            cluster_tumor_stats = {}
            for cluster_id in leiden_clusters:
                cluster_mask = adata_gene.obs['leiden'] == cluster_id
                cluster_tumor_ratios = adata_gene.obs['tumor_ratio'][cluster_mask].values.astype(float)
                cluster_avg_tumor_ratio = np.nanmean(cluster_tumor_ratios) if len(cluster_tumor_ratios) > 0 else np.nan
                cluster_size = cluster_mask.sum()
                cluster_tumor_stats[cluster_id] = {
                    'avg_tumor_ratio': cluster_avg_tumor_ratio,
                    'size': cluster_size,
                    'indices': np.where(cluster_mask)[0]
                }
              
            start_cluster_id = select_root_cluster_simple(cluster_tumor_stats, sample)
            start_cluster_stats = cluster_tumor_stats[start_cluster_id]
            metrics["start_cluster_id"] = start_cluster_id
            metrics["start_cluster_tumor_ratio"] = start_cluster_stats['avg_tumor_ratio']
            metrics["start_cluster_size"] = start_cluster_stats['size']

            start_cluster_indices = start_cluster_stats['indices']
            start_cluster_tumor_ratios = adata_gene.obs['tumor_ratio'].values[start_cluster_indices]

            zero_tumor_indices = start_cluster_indices[np.where(start_cluster_tumor_ratios == 0)[0]]
                
            if len(zero_tumor_indices) > 0:
                if 'X_diffmap' in adata_gene.obsm and adata_gene.obsm['X_diffmap'].shape[1] > 0:
                    comp_arr = adata_gene.obsm['X_diffmap'][:, 0]
                else:
                    comp_arr = adata_gene.obsm['X_pca'][:, 0]
                    
                chosen_root_idx = zero_tumor_indices[np.argmin(comp_arr[zero_tumor_indices])]
                start_root_tumor_ratios = adata_gene.obs['tumor_ratio'].values[chosen_root_idx ]
            else:
                min_tumor_value = np.nanmin(start_cluster_tumor_ratios)
                min_tumor_indices = start_cluster_indices[np.where(start_cluster_tumor_ratios == min_tumor_value)[0]]  
                if 'X_diffmap' in adata_gene.obsm and adata_gene.obsm['X_diffmap'].shape[1] > 0:
                    comp_arr = adata_gene.obsm['X_diffmap'][:, 0]
                else:
                    comp_arr = adata_gene.obsm['X_pca'][:, 0]
                    
                chosen_root_idx = min_tumor_indices[np.argmin(comp_arr[min_tumor_indices])]
                start_root_tumor_ratios = adata_gene.obs['tumor_ratio'].values[chosen_root_idx ]
            
        else:
            metrics["reason"] = "Missing cell_seg data file"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue

        root_idx = int(chosen_root_idx)

        # 3. Check the connectivity of the adjacency graph
        if 'connectivities' in adata_gene.obsp:
            G_gene = adata_gene.obsp['connectivities']
        else:
            metrics["reason"] = "Missing connectivities in gene data"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue

        n_comp_gene, labels_gene = connected_components(G_gene, directed=False, connection='weak')
        print(f"Number of connected components in the genetic data graph: {n_comp_gene}")

        #4. Check the single connected component
        if n_comp_gene != 1:
            metrics["reason"] = f"Gene graph not connected ({n_comp_gene} components)"
            print(f"[Skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue

        if seg_available:
            tumor_ratio = adata_gene.obs['tumor_ratio'].values.astype(float)
            metrics["tumor_ratio_mean"] = np.mean(tumor_ratio)
            metrics["tumor_ratio_max"] = np.max(tumor_ratio)
            metrics["tumor_ratio_median"] = np.median(tumor_ratio)

        x_coords, y_coords, _ = get_spatial_coords(adata_gene, sample)
        if 0 <= root_idx < len(x_coords):
            root_coord = (float(x_coords[root_idx]), float(y_coords[root_idx]))

            adata_gene.uns['iroot'] = root_idx
            adata_img.uns['iroot'] = root_idx            
            root_spot_name = adata_gene.obs_names[root_idx]
            print(f"Root node information: {root_spot_name}, index: {root_idx}")
            print(f"Root node coordinates: ({root_coord[0]:.2f}, {root_coord[1]:.2f})")
        else:
            raise ValueError(f"Invalid root node index: {root_idx}, out of valid range")

        # 5. Calculate pseudo-time of genes
        try:
            sc.tl.dpt(adata_gene, n_dcs=10)
        except TypeError as e:
            try:
                sc.tl.dpt(adata_gene)
                print("Calculate the gene using default parameters DPT")
            except Exception as e2:
                print(f"Genetic DPT calculation failed:{e2}")
                raise
        gene_ptime = adata_gene.obs['dpt_pseudotime'].values.copy()

        #6. Validity of Genomic Temporal Information
        gene_ptime_valid, gene_nan_count, gene_inf_count = check_pseudotime_validity_strict(
            gene_ptime, sample, "gene"
        )
        if not gene_ptime_valid:
            metrics["reason"] = f"Gene pseudotime invalid (NaN:{gene_nan_count}, INF:{gene_inf_count})"
            print(f"[skip] {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue

        gene_ptime_clean = gene_ptime[np.isfinite(gene_ptime)]
        metrics["gene_ptime_mean"] = np.mean(gene_ptime_clean)
        metrics["gene_ptime_std"] = np.std(gene_ptime_clean)
        metrics["gene_ptime_min"] = np.min(gene_ptime_clean)
        metrics["gene_ptime_max"] = np.max(gene_ptime_clean)
        if cell_composition_df is not None and not cell_composition_df.empty:
            spot_cell_summary = cell_composition_df.groupby(['spot_barcode', 'cell_type']).size().unstack(fill_value=0)
            all_spot_barcodes = adata_gene.obs_names
            all_cell_types = cell_composition_df['cell_type'].unique()
            cell_type_counts_df = pd.DataFrame(index=all_spot_barcodes, columns=all_cell_types, data=0)
            cell_type_counts_df.update(spot_cell_summary)
            cell_type_counts_df['total_cells'] = cell_type_counts_df.sum(axis=1)
        else:
            cell_type_counts_df = pd.DataFrame(index=adata_gene.obs_names)
            cell_type_counts_df['total_cells'] = 0

        gene_ptime_df = pd.DataFrame({
            "barcode": adata_gene.obs_names,
            "gene_pseudotime": gene_ptime,
            "tumor_ratio": adata_gene.obs.get('tumor_ratio', np.nan),
            "tumor_count": adata_gene.obs.get('tumor_count', np.nan),
            "cell_count": adata_gene.obs.get('cell_count', np.nan),
            "leiden_cluster": adata_gene.obs.get('leiden', np.nan),
            "spatial_x": x_coords,
            "spatial_y": y_coords,
            "is_root": [i == root_idx for i in range(len(gene_ptime))]
        })

        for col in cell_type_counts_df.columns:
            gene_ptime_df[col] = cell_type_counts_df[col].values
        gene_ptime_path = os.path.join(save_dir, f"{sample}_gene_pseudotime_results.csv")
        gene_ptime_df.to_csv(gene_ptime_path, index=False, encoding="utf-8-sig")
        print(f"Gene pseudotime result has been saved:{gene_ptime_path}")

        # =======================Image feature pseudo-time calculation=====================   
        img_ptime = None    

        sc.tl.pca(adata_img, n_comps=N_PCS, random_state=0)
        sc.pp.neighbors(adata_img,  n_pcs=N_PCS, random_state=0)
        if DIFFMAP:
            sc.tl.diffmap(adata_img)
            sc.pp.neighbors(adata_img, use_rep='X_diffmap', random_state=0)
        sc.tl.leiden(adata_img, resolution=RESOLUTION, random_state=0)

        # Check the connectivity of the image data
        img_pseudotime_success = True
       
        if 'connectivities' in adata_img.obsp:
            G_img = adata_img.obsp['connectivities']
            n_comp_img, _ = connected_components(G_img, directed=False, connection='weak')
            print(f"Number of connected components in the image data graph: {n_comp_img}")
            
            if n_comp_img != 1:
                print(f"{sample}: The image data graph is not connected (with {n_comp_img} components)")
                img_pseudotime_success = False
        else:
            print(f"{sample}: The image data lacks connected components")
            img_pseudotime_success = False

        if img_pseudotime_success:
            try:
                sc.tl.dpt(adata_img, n_dcs=10)
                img_ptime = adata_img.obs['dpt_pseudotime'].values.copy()
 
                img_ptime_valid, img_nan_count, img_inf_count = check_pseudotime_validity_strict(
                    img_ptime, sample, "image"
                )
                if not img_ptime_valid:
                    print(f" {sample}: Image pseudo-time is invalid (NaN: {img_nan_count}, INF: {img_inf_count})")
                    img_pseudotime_success = False
                    img_ptime = None
                else:
                    img_ptime_clean = img_ptime[np.isfinite(img_ptime)]
                    metrics["img_ptime_mean"] = np.mean(img_ptime_clean)
                    metrics["img_ptime_std"] = np.std(img_ptime_clean)
                    metrics["img_ptime_min"] = np.min(img_ptime_clean)
                    metrics["img_ptime_max"] = np.max(img_ptime_clean)
                    
            except Exception as e:
                print(f"{sample} Image pseudotime calculation failed:{e}")
                img_pseudotime_success = False
                img_ptime = None

            img_ptime_df = pd.DataFrame({
                "barcode": adata_img.obs_names,
                "image_pseudotime": img_ptime,
                "tumor_ratio": adata_img.obs.get('tumor_ratio', np.nan),
                "tumor_count": adata_img.obs.get('tumor_count', np.nan),
                "cell_count": adata_img.obs.get('cell_count', np.nan),
                "leiden_cluster": adata_img.obs.get('leiden', np.nan),
                "spatial_x": x_coords,
                "spatial_y": y_coords,
                "is_root": [i == root_idx for i in range(len(img_ptime))]
            })
            for col in cell_type_counts_df.columns:
                img_ptime_df[col] = cell_type_counts_df[col].values

            img_ptime_path = os.path.join(save_dir, f"{sample}_image_pseudotime_results.csv")
            img_ptime_df.to_csv(img_ptime_path, index=False, encoding="utf-8-sig")
            print(f"Image pseudotime result has been saved: {img_ptime_path}")
       
            # ===================== Calculation of similarity indicators =====================
            corr_metrics = compute_correlation_metrics(gene_ptime, img_ptime)
            ranking_metrics = compute_ranking_metrics(gene_ptime, img_ptime)
            dist_metrics = compute_distribution_metrics(gene_ptime, img_ptime)
            cos_sim = compute_cosine_similarity(gene_ptime, img_ptime)

            metrics.update(corr_metrics)
            metrics.update(ranking_metrics)
            metrics.update(dist_metrics)
            metrics["cosine_similarity"] = cos_sim

            plot_image_pseudotime_distribution(img_ptime, sample, save_dir)
            adata_img.obs['image_pseudotime'] = img_ptime
            save_spatial_with_root(
                adata_for_plot=adata_img,
                color_key='image_pseudotime',
                root_coord=root_coord,
                save_path=os.path.join(save_dir, f"{sample}_image_pseudotime_spatial.png"),
                sample_name=sample
            )
            adata_img_updated = plot_image_umap_pseudotime(adata_img.copy(), gene_ptime, img_ptime, sample, save_dir)
            if adata_img_updated is not None:
                adata_img = adata_img_updated

        else:
            print(f"{sample}: Image pseudotime is invalid")
    
        metrics["n_valid_spots"] = adata_gene.n_obs

        plot_gene_pseudotime_distribution(gene_ptime, sample, save_dir)
        adata_gene.obs['gene_pseudotime'] = gene_ptime
        save_spatial_with_root(
            adata_for_plot=adata_gene,
            color_key='gene_pseudotime',
            root_coord=root_coord,
            save_path=os.path.join(save_dir, f"{sample}_gene_pseudotime_spatial.png"),
            sample_name=sample
        )
        plot_tumor_ratio_spatial(
            adata_before = adata_before_cell_filter,
            adata_after  = adata_gene,
            sample_name  = sample,
            save_dir     = save_dir,
            sample_dir   = sample_dir
        )

        adata_gene_updated = plot_gene_umap_pseudotime(adata_gene.copy(), gene_ptime, img_ptime if img_ptime is not None else np.full(len(gene_ptime), np.nan), sample, save_dir)
        if adata_gene_updated is not None:
            adata_gene = adata_gene_updated

        df_res = adata_gene.obs[['dpt_pseudotime', 'tumor_ratio', 'tumor_count', 'cell_count', 'leiden']].copy()
        df_res.rename(columns={'dpt_pseudotime': 'gene_pseudotime'}, inplace=True)
        df_res['image_pseudotime'] = img_ptime if img_ptime is not None else np.nan
        df_res['spot_barcode'] = adata_gene.obs_names.values
        df_res['is_root'] = [i == root_idx for i in range(len(gene_ptime))]
        
        x_coords, y_coords, _ = get_spatial_coords(adata_gene, sample)
        df_res['spatial_x'] = x_coords
        df_res['spatial_y'] = y_coords

        for col in [
            'pearson_r', 'pearson_p', 'kendall_tau', 'kendall_p', 'spearman_r', 'spearman_p',
            'distance_corr', 'kendall_tau_b', 'kendall_tau_b_p', 'tau_distance',
            'concordant_pairs', 'discordant_pairs', 'wasserstein_distance', 'cosine_similarity',
            'tumor_ratio_mean', 'tumor_ratio_max', 'tumor_ratio_median',
            'cell_count_mean', 'cell_count_max', 'cell_count_median',
            'zero_cell_spots', 'zero_cell_ratio',
            'total_cells_matched', 'unique_cell_types', 'avg_cells_per_spot',
            'final_cell_threshold', 'leiden_clusters', 'start_cluster_id',
            'start_cluster_tumor_ratio', 'start_cluster_size',
            'original_spots', 'after_qc', 'after_tissue_filter', 'after_cell_filter', 'after_all_filters',
            'retention_rate_qc', 'retention_rate_tissue', 'retention_rate_cell', 'retention_rate_total'
        ]:
            df_res[col] = metrics[col]

        df_res.to_csv(os.path.join(save_dir, f"{sample}_detailed_results.csv"), index=False, encoding="utf-8-sig")

        metrics_summary = pd.DataFrame({col: [metrics[col]] for col in summary_columns if col != "sample"})
        metrics_summary.to_csv(os.path.join(save_dir, f"{sample}_metrics_summary.csv"), index=False, encoding="utf-8-sig")
        print(f"The summary table of individual sample indicators has been saved")
        summary_rows.append([metrics[col] for col in summary_columns])

    except Exception as e:
        metrics["reason"] = f"Processing error: {str(e)[:100]}"
        print(f" Processing of sample {sample} failed: {e}")
        import traceback
        traceback.print_exc()
        summary_rows.append([metrics[col] for col in summary_columns])

summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
summary_path = os.path.join(SAVE_ROOT, "summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"\nAll sample processing is completed :{summary_path}")
print(f"Number of samples successfully processed: {sum(summary_df['reason'] == 'OK')}/{len(samples)}")