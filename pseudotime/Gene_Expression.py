import os
import json
import warnings
import mygene
import re
import time
import h5py
import requests 
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from bs4 import BeautifulSoup
from functools import lru_cache
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from shapely.geometry import Point, Polygon, MultiPolygon, shape as shapely_shape
from shapely.ops import unary_union
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

# ==================== Parameter configuration ====================
#Sample Data Directory
ROOT_DIR = "samples"
#Result Saving Directory
OUTPUT_DIR = "gene_expression_PT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SPOTS = 100 
MIN_GENES_PER_CELL = 100  
MIN_CELLS_PER_GENE = 3  
MAX_MT_PERCENT = 20  

# The file containing the list of samples to be analyzed
SAMPLE_LIST_FILE = "samples.txt"
# The file containing the list of genes to be analyzed
GENE_LIST_FILE = "genes.txt"
# ENSG mapping file
ENSG_MAPPING_FILE = "ENSG_mapping.txt"

# The tissues that require gene mapping
MAPPING_TISSUES = ['breast', 'Breast', 'BREAST']

USE_TISSUE_CONTOUR = True  
USE_CELL_FILTER = True
CELLVIT_DIRNAME = "cellvit_seg"
CELLVIT_FILENAME_TEMPLATE = "{sample}_cellvit_seg.geojson"

# Genetic mapping cache file
MYGENE_CACHE_FILE = os.path.join(OUTPUT_DIR, "mygene_cache.json")
UNMAPPED_GENES_FILE = os.path.join(OUTPUT_DIR, "unmapped_genes.txt")
REQUEST_DELAY = 0.5

# Ensembl GRCh37 configure
ENSEMBL_GRCH37_URL = "https://grch37.ensembl.org"
ENSEMBL_LOOKUP_URL = f"{ENSEMBL_GRCH37_URL}/Homo_sapiens/Gene/Summary"

# Configure the requests session
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
mg = mygene.MyGeneInfo()

plt.switch_backend('Agg')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 50
plt.rcParams['axes.titlesize'] = 50
plt.rcParams['axes.labelsize'] = 50
plt.rcParams['xtick.labelsize'] = 50
plt.rcParams['ytick.labelsize'] = 50
plt.rcParams['legend.fontsize'] = 50

SPOT_SIZE = 1.3 
CMAP = 'hot'
WSI_ALPHA = 0.6 
SPOT_ALPHA =1 
SPOT_EDGECOLOR = 'black'  
SPOT_LINEWIDTH = 0.5  
# ==================== Load the ENSG mapping file ====================
def load_ensg_mapping(mapping_file):
    symbol_to_ensg = {} 
    ensg_to_symbol = {}  
    
    if not os.path.exists(mapping_file):
        print(f"The ENSG mapping file does not exist: {mapping_file}")
        return symbol_to_ensg, ensg_to_symbol
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    gene_symbol = parts[0].strip()
                    ensg_id = parts[1].strip()
                    base_ensg = ensg_id.split('.')[0] if ensg_id.startswith('ENSG') else ensg_id
                    symbol_upper = gene_symbol.upper()
                    symbol_to_ensg[symbol_upper] = base_ensg
                    ensg_to_symbol[base_ensg] = gene_symbol
                    ensg_to_symbol[ensg_id] = gene_symbol
                    
        print(f" {mapping_file} loading {len(symbol_to_ensg)} gene mapping relationships")
         
    except Exception as e:
        print(f" Failed to read the ENSG mapping file:{e}")
        import traceback
        traceback.print_exc()
    
    return symbol_to_ensg, ensg_to_symbol

SYMBOL_TO_ENSG_MAP, ENSG_TO_SYMBOL_MAP = load_ensg_mapping(ENSG_MAPPING_FILE)

# ==================== Merge gene name processing function ====================
def parse_merged_gene_id(merged_id):
    match = re.match(r'__ambiguous\[(.*?)\]', merged_id)
    if match:
        content = match.group(1)
        parts = [p for p in content.split('+') if p.strip()]
        return parts
    return [merged_id]

def is_merged_gene_id(gene_id):
    return bool(re.match(r'__ambiguous\[.*?\]', str(gene_id)))

def load_cache():
    """Load the cache of mygene"""
    if os.path.exists(MYGENE_CACHE_FILE):
        try:
            with open(MYGENE_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                print(f"Load from cache {len(cache)} gene mapping")
                return cache
        except Exception as e:
            print(f"Failed to read the cache file:{e}")
    return {}

def save_cache(cache):
    try:
        with open(MYGENE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Cache has been saved to:{MYGENE_CACHE_FILE}")
    except Exception as e:
        print(f"Failed to save cache file:{e}")

# ==================== Gene Symbol mapping function ====================
GENE_SYMBOL_CACHE = load_cache()

def normalize_gene_name(gene_name):
    if gene_name is None:
        return None
    return gene_name.upper()

@lru_cache(maxsize=10000)
def query_ensembl_grch37_symbol(ensembl_id):
    """Query the gene symbol from Ensembl GRCh37"""
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    params = {'g': base_id}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        print(f"Query Ensembl GRCh37: {base_id}")
        response = session.get(ENSEMBL_LOOKUP_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('title')
            if title_tag and 'Summary' in title_tag.text:
                title_text = title_tag.text
                match = re.search(r'Gene:\s+([^\s]+)', title_text)
                if match:
                    symbol = match.group(1)
                    return symbol
            
            h1_tag = soup.find('h1')
            if h1_tag:
                h1_text = h1_tag.text
                match = re.search(r'Gene:\s+([^\s]+)', h1_text)
                if match:
                    symbol = match.group(1)
                    return symbol
            
            gene_name_span = soup.find('span', class_='gene-name')
            if gene_name_span:
                symbol = gene_name_span.text.strip()
                return symbol
            
            print(f"Not found in Ensembl GRCh37:{base_id}")
        
        elif response.status_code == 429:
            print(f"Trigger current limiting and wait for a longer period of time...")
            time.sleep(5)
            return None
        else:
            print(f"Ensembl GRCh37 returns a status code{response.status_code}: {base_id}")
        
        return None
        
    except Exception as e:
        print(f"Ensembl GRCh37查询失败 {base_id}: {e}")
        return None
    finally:
        time.sleep(REQUEST_DELAY)

@lru_cache(maxsize=10000)
def query_mygene_gene_symbol(ensembl_id):
    """Query the gene symbol from mygene"""
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    
    try:
        print(f"Query mygene: {base_id}")
        result = mg.query(base_id, scopes='ensembl.gene', fields='symbol', species='human')
        
        if result and 'hits' in result and len(result['hits']) > 0:
            hit = result['hits'][0]
            if 'symbol' in hit:
                symbol = hit['symbol']
                print(f"Find:{base_id} -> {symbol}")
                return symbol
        
        print(f"Not Find:{base_id}")
        return None
            
    except Exception as e:
        print(f"mygene query failed:{ensembl_id}: {e}")
        return None

def get_gene_symbol_from_sources(ensembl_id):
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    
    # 1. Check the cache
    if base_id in GENE_SYMBOL_CACHE:
        return GENE_SYMBOL_CACHE[base_id]
    if ensembl_id in GENE_SYMBOL_CACHE:
        return GENE_SYMBOL_CACHE[ensembl_id]
    
    # 2. Check the ENSG mapping file (find the symbol from ENSG)
    if base_id in ENSG_TO_SYMBOL_MAP:
        symbol = ENSG_TO_SYMBOL_MAP[base_id]
        GENE_SYMBOL_CACHE[base_id] = symbol
        GENE_SYMBOL_CACHE[ensembl_id] = symbol
        print(f"Found from the mapping file:{base_id} -> {symbol}")
        return symbol
    
    # 3.Query mygene
    symbol = query_mygene_gene_symbol(base_id)
    if symbol and symbol != base_id:
        GENE_SYMBOL_CACHE[base_id] = symbol
        GENE_SYMBOL_CACHE[ensembl_id] = symbol
        return symbol
    
    # 4. Query Ensembl GRCh37
    symbol = query_ensembl_grch37_symbol(base_id)
    if symbol:
        GENE_SYMBOL_CACHE[base_id] = symbol
        GENE_SYMBOL_CACHE[ensembl_id] = symbol
        return symbol
    
    return None

def get_ensg_id_from_symbol(gene_symbol):
    if gene_symbol is None:
        return None
    symbol_upper = gene_symbol.upper()
    
    # 1. Check SYMBOL_TO_ENSG_MAP
    if symbol_upper in SYMBOL_TO_ENSG_MAP:
        ensg_id = SYMBOL_TO_ENSG_MAP[symbol_upper]
        print(f"Found from the mapping file:{gene_symbol} (match {symbol_upper}) -> {ensg_id}")
        return ensg_id
    
    # 2. Query mygene
    try:
        print(f"Query mygene: {gene_symbol}")
        result = mg.query(gene_symbol, scopes='symbol', fields='ensembl.gene', species='human')
        if result and 'hits' in result and len(result['hits']) > 0:
            hit = result['hits'][0]
            if 'ensembl' in hit and 'gene' in hit['ensembl']:
                ensg_id = hit['ensembl']['gene']
        
                GENE_SYMBOL_CACHE[ensg_id] = gene_symbol
                print(f"Find:{gene_symbol} -> {ensg_id}")
                return ensg_id
    except Exception as e:
        print(f"Mygene query failed{gene_symbol}: {e}")
    
    return None

def query_merged_gene_symbol(merged_id):
    parts = parse_merged_gene_id(merged_id)
    if not parts:
        return merged_id
    
    print(f"Handle the merging of gene names: {len(parts)} ENSG ID")
    
    converted_parts = []
    for i, part in enumerate(parts):
        print(f"    [{i+1}/{len(parts)}] query: {part}")
        symbol = get_gene_symbol_from_sources(part)
        if symbol and symbol != part:
            converted_parts.append(symbol)
            print(f"Find: {symbol}")
        else:
            converted_parts.append(part)
            print(f"Query failed. Keep the original Gene ID")
        
        if (i + 1) % 5 == 0 and i + 1 < len(parts):
            time.sleep(1)
    
    result = '/'.join(converted_parts)
    print(f"Combined gene conversion results:{result}")
    return result

def convert_gene_list_for_breast(gene_list):
    if not gene_list:
        return {}, {}, []
    
    unique_genes = list(set(gene_list))
    # Establish a two-way mapping
    symbol_to_ensg = {}  
    ensg_to_symbol = {} 
    unmapped = []
    
    for gene in unique_genes:
        # If the input is already an ENSG number, use it directly
        if gene.startswith('ENSG'):
            ensg_to_symbol[gene] = gene
            symbol_to_ensg[gene] = gene
            continue
        
        # Query the ENSG number corresponding to the gene symbol
        ensg_id = get_ensg_id_from_symbol(gene)
        if ensg_id:
            ensg_to_symbol[ensg_id] = gene
            symbol_to_ensg[gene] = ensg_id
            print(f"{gene} -> {ensg_id}")
        else:
            unmapped.append(gene)
            print(f"{gene} No ENSG number was found.")
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\n Successful mappings:{len(ensg_to_symbol)} ")
    if unmapped:
        print(f"Not mapped:{len(unmapped)} ")
    
    return symbol_to_ensg, ensg_to_symbol, unmapped

# ==================== Read the input file ====================
def read_sample_list(file_path):
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    samples.append(line)
        print(f" {file_path} Reading Samples: {len(samples)}")
        return samples
    except Exception as e:
        print(f"Failed to read the sample list file: {e}")
        return []

def read_gene_list(file_path):
    genes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    genes.append(line)
        print(f" {file_path} Reading Genes: {len(genes)}")
        return genes
    except Exception as e:
        print(f"Failed to read the gene list file:{e}")
        return []

# ==================== Obtain the type of sample tissue ====================
def normalize_tissue_name(tissue_name):
    """Standardized organization name"""
    if not tissue_name or str(tissue_name).lower() == "unknown":
        return "Unknown"
    
    tissue_lower = str(tissue_name).strip().lower()
    
    if 'breast' in tissue_lower:
        return 'Breast'
    
    tissue_mapping = {
        'brain': 'Brain', 
        'colon': 'Colon',
        'prostate': 'Prostate',
        'skin': 'Skin', 
        'lymph node': 'Lymph',
    }
    
    for key, value in tissue_mapping.items():
        if key in tissue_lower:
            return value
    words = tissue_lower.split()
    return words[0].capitalize() if words else "Unknown"

def get_sample_tissue(sample_name):
    """The type of tissue from which the samples were obtained"""
    metadata_path = os.path.join(ROOT_DIR, sample_name, "metadata", f"{sample_name}.json")
    if not os.path.exists(metadata_path):
        return "Unknown"
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        tissue_fields = [
            'tissue', 'tissue_type', 'tissue_origin', 'organ',
            'organ_type', 'sample_tissue', 'primary_site', 'site',
            'disease_type', 'tumor_tissue_site', 'tissue_of_origin'
        ]
        
        raw_tissue = None
        for field in tissue_fields:
            if field in metadata:
                raw_tissue = metadata[field]
                break
        
        if raw_tissue is None:
            return "Unknown"
        
        return normalize_tissue_name(raw_tissue)
        
    except Exception as e:
        return "Unknown"

def is_breast_sample(sample_name):
    tissue = get_sample_tissue(sample_name)
    return tissue.lower() in ['breast', 'breast', 'breast']

# ==================== Obtain the spatial coordinates of the spot====================
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
    
    raise KeyError(f"[{sample_name}] No space coordinates were found. The coordinates either do not exist or are invalid")

# ==================== Read the outline of the tissue ====================
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
            print(f"[Error] Unable to read geojson：{e} / {e2}")
            return None, geojson_path
# ==================== Read the cell segmentation data ====================
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
    return cell_polygons, cell_types, fname
# ==================== Obtain the resolution of the sample pixels ====================
def get_sample_pixel_resolution(sample_name, sample_dir):
    """Obtain the resolution from the metadata file"""
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    if not os.path.exists(metadata_path):
        print(f"[{sample_name}] Metadata file not found {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'pixel_size_um_estimated' in metadata:
            pixel_resolution = float(metadata['pixel_size_um_estimated'])
            return pixel_resolution
        else:
            return None
    except Exception as e:
        return None

def get_real_spot_pixel_radius(sample_name, sample_dir):
    """Obtain the actual pixel radius of the Spot"""
    pixel_resolution = get_sample_pixel_resolution(sample_name, sample_dir)
    if pixel_resolution is None:
        return None
    
    # Obtain the physical diameter from the metadata
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'spot_diameter' in metadata:
                spot_physical_diam = float(metadata['spot_diameter'])
        except Exception as e:
            spot_physical_diam = 55  
    else:
        spot_physical_diam = 55
    
    # Calculate the radius
    spot_pixel_diam = spot_physical_diam / pixel_resolution
    spot_pixel_radius = spot_pixel_diam / 2.0
    return spot_pixel_radius
# ==================== Organ contour filtering ====================
def filter_spots_by_tissue(adata, sample, sample_dir, save_dir=None):
    """Filtering spots outside the tissue"""
    original_spots = adata.n_obs
    poly, poly_path = load_polygon_for_sample(sample_dir, sample)
    if poly is None:
        print(f"{poly_path}no organization outline file was found.")
        return adata, np.ones(adata.n_obs, dtype=bool), original_spots, original_spots
    
    # Obtain the coordinates and valid mask
    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample)
    if not np.any(valid_mask):
        print("Without valid coordinates, organization filtering cannot be performed")
        return adata, valid_mask, original_spots, original_spots
    
    # Obtain the physical radius of the spot
    spot_r = get_real_spot_pixel_radius(sample, sample_dir)
    if spot_r is None:
        spot_r = 50  
    
    # Internal judgment logic within the organization
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
            inside_mask[i] = False

    n_total = len(inside_mask)
    n_inside = int(inside_mask.sum())
    n_filtered = n_total - n_inside
    retention_rate = n_inside / n_total if n_total > 0 else 0
    print(f"Organizational filtering result: Total spots {n_total} Reserved spots {n_inside} Filtered spots{n_filtered}")

    filtered_adata = adata[inside_mask].copy()
    return filtered_adata, inside_mask, original_spots, filtered_adata.n_obs

# ==================== Cell count filtration ====================
def filter_spots_by_cell_count_enhanced(adata, sample_name=""):

    if 'cell_count' not in adata.obs.columns:
        print(f"[{sample_name}] no 'cell_count' data in the adata, so skip the cell count filtering.")
        mask = np.ones(adata.n_obs, dtype=bool)
        return adata.copy(), mask, None, 0

    # Obtain the number of cells
    cell_counts = adata.obs['cell_count'].astype(int).values
    # Construct the retention mask
    mask_keep = (cell_counts > 0)
    n_total = len(mask_keep)
    n_kept = int(mask_keep.sum())
    n_filtered = n_total - n_kept
    print(f"[{sample_name}] Cell count filtration: Total spots {n_total} Reserved spots {n_kept} Filtered spots{n_filtered}")
    
    filtered_adata = adata[mask_keep].copy()
    final_threshold = 1
    return filtered_adata, mask_keep, cell_counts, final_threshold

# ==================== Count the number of cells ====================
def compute_spot_cell_counts_from_seg_fast(adata, cell_polygons, cell_types, sample_name, sample_dir):

    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample_name)
    n_total_spots = len(x_coords)
    if not valid_mask.any():
        return np.zeros(n_total_spots, int)

    spot_r = get_real_spot_pixel_radius(sample_name, sample_dir)
    if spot_r is None:
        print(f"[{sample_name}] Unable to obtain the spot radius, skip the cell counting")
        return np.zeros(n_total_spots, int)
    
    # Construct Spot GeoDataFrame
    spot_geoms = []
    spot_original_idx = []
    for idx in range(n_total_spots):
        if not valid_mask[idx]:
            continue
        x, y = x_coords[idx], y_coords[idx]
        spot_geoms.append(Point(x, y).buffer(spot_r))
        spot_original_idx.append(idx)
    
    if len(spot_original_idx) == 0:
        return np.zeros(n_total_spots, int)
    
    spots_gdf = gpd.GeoDataFrame({
        'spot_original_idx': spot_original_idx
    }, geometry=spot_geoms, crs="EPSG:4326").rename_geometry('geometry_spot')

    valid_cell_data = []
    for cid, geom, ctype in zip(range(len(cell_polygons)), cell_polygons, cell_types):
        if geom is None or ctype is None:
            continue
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue
        try:
            if not geom.is_valid or geom.area == 0:
                continue
        except Exception:
            continue
        valid_cell_data.append({
            'cell_id': int(cid),
            'cell_type': str(ctype).strip().lower(),
            'geometry_cell': geom
        })
    
    if not valid_cell_data:
        return np.zeros(n_total_spots, int)
    
    cells_geoms = [d['geometry_cell'] for d in valid_cell_data]
    cells_gdf = gpd.GeoDataFrame(valid_cell_data, geometry=cells_geoms, 
                                crs="EPSG:4326").rename_geometry('temp_cell_geom')
    
    # Spatial connection
    try:
        join_res = gpd.sjoin(spots_gdf, cells_gdf, how='inner', predicate='intersects')
    except Exception as e:
        return np.zeros(n_total_spots, int)
    
    # Calculate the number of cells for each spot
    total_count = np.zeros(n_total_spots, int)
    spot_cell_stats = join_res.groupby('spot_original_idx')['cell_id'].nunique()
    for spot_idx, count in spot_cell_stats.items():
        total_count[spot_idx] = int(count)
    
    return total_count
# ==================== QC filter ====================
def perform_qc_filtering(adata):

    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    
    # Mitochondrial gene filtering
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MT_PERCENT].copy()
    
    return adata
# ==================== Obtain gene expression data ====================
def get_gene_expression(adata, gene_name, ensg_to_symbol=None, is_breast=False):
    """Obtain the expression level of the specified gene from adata"""
    var_names = adata.var_names.tolist()
    var_names_lower = {name.lower(): name for name in var_names}
    var_names_upper = {name.upper(): name for name in var_names}
    
    if is_breast and ensg_to_symbol:
        for ensg, symbol in ensg_to_symbol.items():
            if symbol.lower() == gene_name.lower() and ensg in var_names:
                gene_idx = var_names.index(ensg)
                if hasattr(adata.X, 'toarray'):
                    return adata.X[:, gene_idx].toarray().flatten(), ensg
                else:
                    return adata.X[:, gene_idx].flatten(), ensg
    
    gene_lower = gene_name.lower()
    gene_upper = gene_name.upper()
    
    if gene_lower in var_names_lower:
        original_name = var_names_lower[gene_lower]
        gene_idx = var_names.index(original_name)
        if hasattr(adata.X, 'toarray'):
            return adata.X[:, gene_idx].toarray().flatten(), original_name
        else:
            return adata.X[:, gene_idx].flatten(), original_name
    
    if gene_upper in var_names_upper:
        original_name = var_names_upper[gene_upper]
        gene_idx = var_names.index(original_name)
        if hasattr(adata.X, 'toarray'):
            return adata.X[:, gene_idx].toarray().flatten(), original_name
        else:
            return adata.X[:, gene_idx].flatten(), original_name
    
    if gene_name in var_names:
        gene_idx = var_names.index(gene_name)
        if hasattr(adata.X, 'toarray'):
            return adata.X[:, gene_idx].toarray().flatten(), gene_name
        else:
            return adata.X[:, gene_idx].flatten(), gene_name
    
    return None, None

def process_sample(sample, target_genes, ensg_to_symbol, symbol_to_ensg):
    """Process a single sample"""
    print(f"\n{'='*60}")
    print(f"Process sample:{sample}")
    print(f"{'='*60}")

    is_breast = is_breast_sample(sample)
    print(f"  Tissue type: {'Breast' if is_breast else 'Non-breast'}")
    print(f"  Gene Mapping: {'Enable' if is_breast else 'Disable'}")
    
    sample_dir = os.path.join(ROOT_DIR, sample)
    sample_output_dir = os.path.join(OUTPUT_DIR, sample)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    st_path = os.path.join(sample_dir, "st", f"{sample}.h5ad")
    if not os.path.exists(st_path):
        print(f"Data file not found: {st_path}")
        return None
    
    try:
        adata_original = sc.read_h5ad(st_path)
        print(f"    Original data: {adata_original.n_obs} spots  {adata_original.n_vars} genes")
    except Exception as e:
        print(f" Failed to read the expression data: {e}")
        return None
    

    print(f"\nQC filter")
    try:
        adata_qc = perform_qc_filtering(adata_original.copy())
        print(f"QC filter: {adata_qc.n_obs} spots")
    except Exception as e:
        print(f"QC filter failed: {e}")
        return None
    
    if USE_TISSUE_CONTOUR:
        print(f"\n tissue contour filter")
        try:
            adata_tissue, inside_mask, before_tissue, after_tissue = filter_spots_by_tissue(
                adata_qc, sample, sample_dir, sample_output_dir
            )
            print(f"tissue contour filter: {adata_tissue.n_obs} spots")
        except Exception as e:
            print(f"tissue contour filter failed: {e}")
            return None
    else:
        adata_tissue = adata_qc.copy()
        print(f"\n Skip the organization outline filtering")
    

    if USE_CELL_FILTER:
        print(f"\nCell count filtration")

        cell_polygons, cell_types, _ = load_cellvit_polygons(sample_dir, sample)
        if cell_polygons is None or len(cell_polygons) == 0:
            print(f"Cell division data is either absent or empty")
            return None
        
        cell_counts = compute_spot_cell_counts_from_seg_fast(
            adata_tissue, cell_polygons, cell_types, sample, sample_dir
        )
        adata_tissue.obs['cell_count'] = cell_counts

        adata_final, cell_count_mask, counts, threshold = filter_spots_by_cell_count_enhanced(
            adata_tissue, sample
        )
        print(f"Cell count filtration: {adata_final.n_obs} spots")
    else:
        adata_final = adata_tissue.copy()
        print(f"\n Skip Cell count filtration")
        
    n_spots = adata_final.n_obs
    if n_spots < MIN_SPOTS:
        print(f"Final inal effective spots: {n_spots} Less than the minimum requirement")
        return None
    print(f"\n Final inal effective spots: {n_spots}")

    try:
        if hasattr(adata_final.X, 'toarray'):
            adata_final.layers['counts'] = adata_final.X.copy()
        else:
            adata_final.layers['counts'] = adata_final.X

        x_coords, y_coords, valid_mask = get_spatial_coords(adata_final, sample)
        adata_final.obsm['spatial'] = np.vstack([x_coords, y_coords]).T
        
        try:
            adata_final = st.convert_scanpy(adata_final, use_quality='downscaled_fullres').copy()
        except Exception as e:
            if 'spatial' not in adata_final.uns:
                adata_final.uns['spatial'] = {}
            if 'ST' not in adata_final.uns['spatial']:
                adata_final.uns['spatial']['ST'] = {
                    'images': {},
                    'scalefactors': {'tissue_downscaled_fullres_scalef': 1.0, 'spot_diameter_fullres': 50}
                }
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None
    
    results = []
    var_names = adata_final.var_names.tolist()

    if hasattr(adata_final.X, 'toarray'):
        X = adata_final.X.toarray()
    else:
        X = adata_final.X

    expression_matrix = np.zeros((n_spots, len(target_genes)))
    gene_found_mask = np.zeros(len(target_genes), dtype=bool)
    
    for i, gene_symbol in enumerate(target_genes):
        expression, used_gene_id = get_gene_expression(adata_final, gene_symbol, ensg_to_symbol if is_breast else None, is_breast)
        
        if expression is not None:
            expression_matrix[:, i] = expression
            gene_found_mask[i] = True
            
            results.append({
                'gene': gene_symbol,  
                'ensg_id': used_gene_id, 
                'matched_name': gene_symbol,  
                'expression': expression,
                'non_zero_count': np.sum(expression > 0),
                'non_zero_ratio': np.sum(expression > 0) / len(expression) if len(expression) > 0 else 0,
                'found': True
            })
            print(f"Find Gene: {gene_symbol} (Use ID: {used_gene_id}, Non-zero proportion:{np.sum(expression > 0)/len(expression):.2%})")
        else:
            results.append({
                'gene': gene_symbol,
                'ensg_id': None,
                'matched_name': gene_symbol,
                'expression': None,
                'non_zero_count': 0,
                'non_zero_ratio': 0,
                'found': False
            })
            print(f"Not Found Gene: {gene_symbol}")

    found_genes = sum(1 for r in results if r['found'])
    print(f"\n Find Genes:{found_genes}/{len(target_genes)} ")
    
    if found_genes == 0:
        print(f"No target genes were found")
        return None
        
    return {
        'sample': sample,
        'is_breast': is_breast,
        'adata': adata_final,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'barcodes': adata_final.obs_names,
        'results': results,
        'n_found_genes': found_genes,
        'n_spots': n_spots,
        'filtering_stats': {
            'original_spots': adata_original.n_obs,
            'after_qc': adata_qc.n_obs,
            'after_tissue': adata_tissue.n_obs if USE_TISSUE_CONTOUR else adata_qc.n_obs,
            'after_cell': adata_final.n_obs
        }
    }
# ==================== Visual function ====================
def plot_expression_spatial(sample_data, values, color_key, save_path, sample_name, value_name="Expression"):
    adata = sample_data['adata'].copy()
    safe_color_key = re.sub(r'[^a-zA-Z0-9_]', '_', color_key)

    is_gene_plot = False
    ensg_id = None
    for result in sample_data['results']:
        if result['matched_name'] == color_key and result['found']:
            is_gene_plot = True
            if 'ensg_id' in result and result['ensg_id'] is not None:
                ensg_id = result['ensg_id']
            break
    
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    
    try:
        if is_gene_plot and ensg_id and ensg_id in adata.var_names:
            sc.pl.spatial(
                adata,
                color=[ensg_id],
                img_key="downscaled_fullres",
                size=SPOT_SIZE,
                alpha_img=WSI_ALPHA,  
                ncols=1,
                show=False,
                cmap=CMAP,
                title='',  
                ax=ax
            )
        else:
            adata.obs[safe_color_key] = values
            sc.pl.spatial(
                adata,
                color=[safe_color_key],
                img_key="downscaled_fullres",
                size=SPOT_SIZE,
                alpha_img=WSI_ALPHA,  
                ncols=1,
                show=False,
                cmap=CMAP,
                title='',  
                ax=ax
            )
        spatial_success = True
    except Exception as e:
        print(f"sc.pl.spatial Drawing failed: {e}")
        spatial_success = False
    
    if not spatial_success:
        ax.scatter(sample_data['x_coords'], sample_data['y_coords'], 
                  c=values, s=30, cmap=CMAP, alpha=SPOT_ALPHA, 
                  edgecolors=SPOT_EDGECOLOR, linewidths=SPOT_LINEWIDTH)
        ax.invert_yaxis()
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

    if spatial_success:
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

    vals = values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmin == vmax:
        vmin = vmin - 1
        vmax = vmax + 1
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(value_name, fontsize=28)
    cbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('X coordinate', fontsize=28)
    ax.set_ylabel('Y coordinate', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    title = f'{sample_name}-{color_key}'
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_gene_expression_spatial(sample_data, gene_result, output_dir):
    if gene_result['expression'] is None:
        print(f" {gene_result['gene']} no data available, skipping the graph.")
        return False
    
    values = gene_result['expression']
    color_key = gene_result['matched_name']  
    safe_name = color_key.replace('/', '_').replace('\\', '_').replace('*', '_').replace(' ', '_')
    save_path = os.path.join(output_dir, f"{sample_data['sample']}_{safe_name}_expression.png")
    
    plot_expression_spatial(
        sample_data=sample_data,
        values=values,
        color_key=color_key,
        save_path=save_path,
        sample_name=sample_data['sample'],
        value_name="Expression Level"
    )
    return True

def save_expression_csv(sample_data, output_dir):
    sample = sample_data['sample']
    barcodes = sample_data['barcodes']
    
    data = {'barcode': barcodes}
    for gene_result in sample_data['results']:
        if gene_result['found']:
            safe_name = gene_result['matched_name'].replace('/', '_').replace('\\', '_').replace('*', '_').replace(' ', '_')
            data[safe_name] = gene_result['expression']
            if 'ensg_id' in gene_result and gene_result['ensg_id'] is not None:
                data[f"{safe_name}_ENSG"] = gene_result['ensg_id']
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, f"{sample}_expression.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    stats_df = pd.DataFrame([sample_data['filtering_stats']])
    stats_df['is_breast'] = sample_data['is_breast']
    stats_df['n_found_genes'] = sample_data['n_found_genes']
    stats_path = os.path.join(output_dir, f"{sample}_filtering_stats.csv")
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    return csv_path

def save_summary(all_samples_data, output_dir, unmapped_genes=None):
    summary = []
    filtering_summary = []
    
    for sample_data in all_samples_data:
        if sample_data is None:
            continue
        
        for gene_result in sample_data['results']:
            summary.append({
                'sample': sample_data['sample'],
                'is_breast': sample_data['is_breast'],
                'gene': gene_result['gene'],
                'ensg_id': gene_result.get('ensg_id', None),
                'matched_name': gene_result['matched_name'],
                'found': gene_result['found'],
                'non_zero_spots': gene_result['non_zero_count'],
                'non_zero_ratio': gene_result['non_zero_ratio'],
                'total_spots': sample_data['n_spots']
            })
        
        filtering_summary.append({
            'sample': sample_data['sample'],
            'is_breast': sample_data['is_breast'],
            'n_found_genes': sample_data['n_found_genes'],
            'original_spots': sample_data['filtering_stats']['original_spots'],
            'after_qc': sample_data['filtering_stats']['after_qc'],
            'after_tissue': sample_data['filtering_stats']['after_tissue'],
            'after_cell': sample_data['filtering_stats']['after_cell'],
            'final_spots': sample_data['n_spots'],
            'qc_retention': sample_data['filtering_stats']['after_qc'] / sample_data['filtering_stats']['original_spots'],
            'tissue_retention': sample_data['filtering_stats']['after_tissue'] / sample_data['filtering_stats']['after_qc'],
            'cell_retention': sample_data['filtering_stats']['after_cell'] / sample_data['filtering_stats']['after_tissue'],
            'total_retention': sample_data['n_spots'] / sample_data['filtering_stats']['original_spots']
        })
    
    if summary:
        df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, "expression_summary.csv")
        df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n Summary of expressions has been saved: {summary_path}")
    
    if filtering_summary:
        df_filter = pd.DataFrame(filtering_summary)
        filter_path = os.path.join(output_dir, "filtering_summary.csv")
        df_filter.to_csv(filter_path, index=False, encoding='utf-8-sig')
        print(f" Filtering and summarizing has been saved: {filter_path}")

        breast_samples = [f for f in filtering_summary if f['is_breast']]
        non_breast_samples = [f for f in filtering_summary if not f['is_breast']]
        
        
    if unmapped_genes:
        unmapped_path = os.path.join(output_dir, "unmapped_genes.txt")
        with open(unmapped_path, 'w', encoding='utf-8') as f:
            f.write("No matching ENSG number was found \n")
            for gene in unmapped_genes:
                f.write(f"{gene}\n")
        print(f"\nThe list of unmapped genes has been saved:{unmapped_path}")

# ==================== Main function ====================
def main():

    samples = read_sample_list(SAMPLE_LIST_FILE)
    if not samples:
        print("No samples need to be processed")
        return
    
    genes = read_gene_list(GENE_LIST_FILE)
    if not genes:
        print("No genes need to be processed")
        return
     

    breast_samples = []
    non_breast_samples = []
    for sample in samples:
        if is_breast_sample(sample):
            breast_samples.append(sample)
        else:
            non_breast_samples.append(sample)
    
    if breast_samples:
        print(f"Breast samples list: {', '.join(breast_samples)}")
        
    symbol_to_ensg = {}
    ensg_to_symbol = {}
    unmapped_genes = []
    
    if breast_samples:
        print(f"\n{'='*50}")
        print("Perform gene name mapping for the Breast samples:（gene symbol -> ENSG ）")
        print(f"{'='*50}")
        symbol_to_ensg, ensg_to_symbol, unmapped_genes = convert_gene_list_for_breast(genes)
        
        if unmapped_genes:
            print(f"\n Not mapping genes: {len(unmapped_genes)} ")
        
        print(f"\nGenes mapping results:")
        for gene in genes:
            if gene in symbol_to_ensg:
                print(f"  {gene} -> {symbol_to_ensg[gene]}")
            else:
                print(f"  {gene} (Not Mapping)")
    else:
        print(f"\nThere are no samples requiring gene mapping")
        for gene in genes:
            symbol_to_ensg[gene] = gene
            ensg_to_symbol[gene] = gene
    
    all_samples_data = []
    
    for sample in tqdm(samples, desc="Process samples..."):
        sample_data = process_sample(sample, genes, ensg_to_symbol, symbol_to_ensg)
        
        if sample_data is not None:
            all_samples_data.append(sample_data)
            sample_output_dir = os.path.join(OUTPUT_DIR, sample)
            os.makedirs(sample_output_dir, exist_ok=True)
            save_expression_csv(sample_data, sample_output_dir)
            
            print(f"\nDraw a single gene expression map:")
            for gene_result in sample_data['results']:
                if gene_result['found']:
                    plot_gene_expression_spatial(sample_data, gene_result, sample_output_dir)

    save_summary(all_samples_data, OUTPUT_DIR, unmapped_genes)
    if unmapped_genes:
        print(f"unmapped_genes.txt: list of mapped genes was not found")
    print(f"mygene_cache.json:gene mapping cache")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupts the program")
    except Exception as e:
        print(f"\nProgram error occurred:{e}")
        import traceback
        traceback.print_exc()