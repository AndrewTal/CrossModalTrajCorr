import os
import json
import warnings
import time
import gc
import requests
import mygene
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
#Sample Data Directory
ROOT_DIR = "samples"
#Result Saving Directory
PT_DIR = "results/GigaPath_fix"
#Gene Analysis Results
OUTPUT_DIR = "gene_enrich"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pseudotime type selection:'gene'Gene Pseudotime 'image'Image Pseudotime
PSEUDOTIME_TYPE = 'gene'

if PSEUDOTIME_TYPE == 'gene':
    PSEUDOTIME_FILE_SUFFIX = "_gene_pseudotime_results.csv"
    PSEUDOTIME_COLUMN = "gene_pseudotime"
    PSEUDOTIME_NAME = "Gene"
elif PSEUDOTIME_TYPE == 'image':
    PSEUDOTIME_FILE_SUFFIX = "_image_pseudotime_results.csv"
    PSEUDOTIME_COLUMN = "image_pseudotime"
    PSEUDOTIME_NAME = "Image"
else:
    raise ValueError("PSEUDOTIME_TYPE must be either 'gene' or 'image'")

# Mapping configuration
MYGENE_CACHE_FILE = os.path.join(OUTPUT_DIR, "mygene_cache.json")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "mapping.txt")
UNMAPPED_DETAILS_FILE = os.path.join(OUTPUT_DIR, "unmapped_genes_details.txt")
REQUEST_DELAY = 0.5

# Ensembl GRCh37 configuration
ENSEMBL_GRCH37_URL = "https://grch37.ensembl.org"
ENSEMBL_LOOKUP_URL = f"{ENSEMBL_GRCH37_URL}/Homo_sapiens/Gene/Summary"

# Configure requests session
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Filtering parameters
MIN_SAMPLE_FRACTION = 0.5
CORRELATION_THRESHOLD = 0.5
TOP_N_GENES_LIST = [100]
MIN_SPOTS_PER_SAMPLE = 50
N_BINS = 20
BATCH_SIZE = 2000

# Heatmap parameters
DPI = 200
FONT_SIZE = 8
COLORMAP = 'RdBu_r'
MAX_PIXELS = 65000
MIN_EXPRESSION_RATIO = 0.01

# Global variables for unmapped genes
unmapped_genes = defaultdict(int)
unmapped_genes_final = set()
unmapped_genes_details = []

# Initialize mygene client
mg = mygene.MyGeneInfo()

# ==================== Sample list loading ====================
SAMPLE_LIST_FILE = "select_samples.csv" 

def load_samples_from_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Sample list file does not exist: {file_path}")
        return [], {}
    
    try:
        df = pd.read_csv(file_path)
        samples = df['sample'].dropna().astype(str).str.strip().tolist()
        samples = [s for s in samples if s and not s.startswith('#')]
        
        sample_to_tissue = {}
        for _, row in df.iterrows():
            sample = str(row['sample']).strip()
            tissue = str(row['tissue']).strip()
            if sample and not sample.startswith('#'):
                sample_to_tissue[sample] = tissue
        
        print(f"From {os.path.basename(file_path)}, {len(samples)} samples were loaded")
        print(f"Tissue Type: {set(sample_to_tissue.values())}")
        
        return samples, sample_to_tissue
        
    except Exception as e:
        print(f"Failed to read CSV file:{e}")
        return [], {}
    
#The name of the organization to be analyzed. If it is null, then all organizations will be analyzed
TARGET_TISSUES = [] 
# Tissues that don't need gene mapping
NO_MAPPING_TISSUES = ['prostate', 'brain', 'skin(cSCC)', 'skin(Melanoma)','colon', 'lymph node']

SAMPLES, SAMPLE_TO_TISSUE = load_samples_from_csv(SAMPLE_LIST_FILE)
# ==================== Organization name processing function ====================
def get_sample_tissue(sample_name):
    if sample_name in SAMPLE_TO_TISSUE:
        return SAMPLE_TO_TISSUE[sample_name]
    else:
        return "Unknown"

def match_tissue_to_target(tissue_name, target_tissues):
    if not tissue_name or tissue_name == "Unknown":
        return None
    if not target_tissues:
        return tissue_name
    for target in target_tissues:
        if tissue_name == target:
            return target
    
    return None
# ==================== Gene ID Processing Functions ====================
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
    if os.path.exists(MYGENE_CACHE_FILE):
        try:
            with open(MYGENE_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                print(f"  ✓ Loaded {len(cache)} gene mappings from mygene cache")
                return cache
        except Exception as e:
            print(f"Failed to read cache file: {e}")
    else:
        print(f"mygene cache file does not exist")
    return {}

def load_mapping_file():
    mapping = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            ensembl_id = parts[0].strip()
                            gene_symbol = parts[1].strip()
                            mapping[ensembl_id] = gene_symbol
            print(f"Loaded {len(mapping)} gene mappings from mapping.txt")
        except Exception as e:
            print(f"Failed to read mapping.txt: {e}")
    else:
        print(f"mapping.txt file does not exist")
    return mapping

def save_cache(cache):
    try:
        with open(MYGENE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Cache saved to: {MYGENE_CACHE_FILE}")
    except Exception as e:
        print(f"Failed to save cache file: {e}")

def save_unmapped_details():
    if unmapped_genes_details:
        unmapped_genes_details.sort(key=lambda x: x['count'], reverse=True)
        with open(UNMAPPED_DETAILS_FILE, 'w', encoding='utf-8') as f:
            f.write("#" + "="*80 + "\n")
            f.write("# Unmapped Genes Detailed Report\n")
            f.write("# Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("#" + "="*80 + "\n\n")
            f.write(f"Total unmapped genes: {len(unmapped_genes_details)}\n\n")
            f.write("-"*80 + "\n")
            f.write("| No. | ENSG ID | Count | Merged Gene | Note |\n")
            f.write("-"*80 + "\n")
            for i, detail in enumerate(unmapped_genes_details, 1):
                merged_flag = "Yes" if detail['is_merged'] else "No"
                f.write(f"| {i:3d} | {detail['ensembl_id']:<20s} | {detail['count']:6d} | {merged_flag:^6s} | {detail['note']} |\n")
            f.write("-"*80 + "\n\n")
            f.write("\n## Detailed unmapped gene list (sorted by count)\n\n")
            for detail in unmapped_genes_details:
                f.write(f"{detail['ensembl_id']}\t{detail['count']}\t{detail['note']}\n")
        print(f"\nUnmapped genes details saved to: {UNMAPPED_DETAILS_FILE}")

def add_unmapped_detail(ensembl_id, note="", is_merged=False):
    global unmapped_genes_details
    for detail in unmapped_genes_details:
        if detail['ensembl_id'] == ensembl_id:
            detail['count'] = unmapped_genes[ensembl_id]
            return
    unmapped_genes_details.append({
        'ensembl_id': ensembl_id,
        'count': unmapped_genes[ensembl_id],
        'is_merged': is_merged,
        'note': note
    })

# ==================== Gene Symbol Mapping Functions ====================
GENE_SYMBOL_MAP = load_cache()
MAPPING_MAP = load_mapping_file()
GENE_SYMBOL_MAP.update(MAPPING_MAP)
print(f"Total mappings after merging: {len(GENE_SYMBOL_MAP)} genes")

@lru_cache(maxsize=10000)
def query_ensembl_grch37_symbol(ensembl_id):
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    params = {'g': base_id}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        print(f"Querying Ensembl GRCh37: {base_id}")
        response = session.get(ENSEMBL_LOOKUP_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('title')
            if title_tag and 'Summary' in title_tag.text:
                title_text = title_tag.text
                match = re.search(r'Gene:\s+([^\s]+)', title_text)
                if match:
                    symbol = match.group(1)
                    print(f"Found in title: {base_id} -> {symbol}")
                    return symbol
            
            h1_tag = soup.find('h1')
            if h1_tag:
                h1_text = h1_tag.text
                match = re.search(r'Gene:\s+([^\s]+)', h1_text)
                if match:
                    symbol = match.group(1)
                    print(f"Found in h1: {base_id} -> {symbol}")
                    return symbol
            
            gene_name_span = soup.find('span', class_='gene-name')
            if gene_name_span:
                symbol = gene_name_span.text.strip()
                print(f"Found in gene-name: {base_id} -> {symbol}")
                return symbol
            
            print(f"Not found in Ensembl GRCh37: {base_id}")
            add_unmapped_detail(ensembl_id, "Ensembl GRCh37 query returned no results")
        
        elif response.status_code == 429:
            print(f"Rate limit triggered, waiting longer...")
            time.sleep(5)
            add_unmapped_detail(ensembl_id, "Rate limit triggered")
            return None
        else:
            print(f"Ensembl GRCh37 returned status code {response.status_code}: {base_id}")
            add_unmapped_detail(ensembl_id, f"HTTP {response.status_code}")
        
        unmapped_genes[ensembl_id] += 1
        unmapped_genes_final.add(ensembl_id)
        return None
        
    except Exception as e:
        print(f"Ensembl GRCh37 query failed for {base_id}: {e}")
        add_unmapped_detail(ensembl_id, f"Error: {str(e)[:50]}")
        unmapped_genes_final.add(ensembl_id)
        return None
    finally:
        time.sleep(REQUEST_DELAY)

@lru_cache(maxsize=10000)
def query_mygene_gene_symbol(ensembl_id):
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    
    try:
        print(f"Querying mygene: {base_id}")
        result = mg.query(base_id, scopes='ensembl.gene', fields='symbol', species='human')
        
        if result and 'hits' in result and len(result['hits']) > 0:
            hit = result['hits'][0]
            if 'symbol' in hit:
                symbol = hit['symbol']
                print(f"    ✓ Found in mygene: {base_id} -> {symbol}")
                return symbol
        
        print(f"Not found in mygene: {base_id}")
        add_unmapped_detail(ensembl_id, "mygene query returned no results")
        return None
            
    except Exception as e:
        print(f"mygene query failed for {ensembl_id}: {e}")
        add_unmapped_detail(ensembl_id, f"mygene error: {str(e)[:50]}")
        return None

def get_gene_symbol_from_sources(ensembl_id):
    base_id = ensembl_id.split('.')[0] if ensembl_id.startswith('ENSG') else ensembl_id
    
    if base_id in GENE_SYMBOL_MAP:
        return GENE_SYMBOL_MAP[base_id]
    if ensembl_id in GENE_SYMBOL_MAP:
        return GENE_SYMBOL_MAP[ensembl_id]
    
    symbol = query_mygene_gene_symbol(base_id)
    if symbol and symbol != base_id:
        GENE_SYMBOL_MAP[base_id] = symbol
        GENE_SYMBOL_MAP[ensembl_id] = symbol
        return symbol
    
    symbol = query_ensembl_grch37_symbol(base_id)
    if symbol:
        GENE_SYMBOL_MAP[base_id] = symbol
        GENE_SYMBOL_MAP[ensembl_id] = symbol
        return symbol
    
    return None

def query_merged_gene_symbol(merged_id):
    parts = parse_merged_gene_id(merged_id)
    if not parts:
        add_unmapped_detail(merged_id, "Empty merged gene", is_merged=True)
        return merged_id
    
    print(f"Processing merged gene name with {len(parts)} ENSG IDs")
    
    converted_parts = []
    unmapped_parts = []
    
    for i, part in enumerate(parts):
        print(f"[{i+1}/{len(parts)}] Querying: {part}")
        symbol = get_gene_symbol_from_sources(part)
        
        if symbol and symbol != part:
            converted_parts.append(symbol)
            print(f"Found: {symbol}")
        else:
            unmapped_parts.append(part)
            converted_parts.append(part)
            print(f"Query failed, keeping original ID")
            add_unmapped_detail(part, "Unmapped part in merged gene")
        
        if (i + 1) % 5 == 0 and i + 1 < len(parts):
            time.sleep(1)
    
    if unmapped_parts:
        unmapped_count = len(unmapped_parts)
        print(f"Merged gene has {unmapped_count} unmapped ENSG IDs")
        for part in unmapped_parts:
            unmapped_genes[part] += 1
            unmapped_genes_final.add(part)
        add_unmapped_detail(merged_id, f"Contains {unmapped_count} unmapped parts", is_merged=True)
    
    result = '/'.join(converted_parts)
    print(f"Merged gene conversion result: {result}")
    return result

def batch_convert_to_symbols(ensembl_ids, need_mapping=True):
    if not need_mapping:
        print(f"Tissue does not need gene mapping, using original IDs")
        return {ensembl_id: ensembl_id for ensembl_id in ensembl_ids}
    
    if not ensembl_ids:
        return {}
    
    unique_ids = list(set(ensembl_ids))
    print(f"\nStarting batch conversion of {len(unique_ids)} gene IDs...")
    normal_count = sum(1 for id in unique_ids if not is_merged_gene_id(id))
    merged_count = len(unique_ids) - normal_count
    print(f"Normal IDs: {normal_count}, Merged IDs: {merged_count}")
    
    result_map = {}
    
    normal_ids = [id for id in unique_ids if not is_merged_gene_id(id)]
    for ensembl_id in normal_ids:
        symbol = get_gene_symbol_from_sources(ensembl_id)
        if symbol:
            result_map[ensembl_id] = symbol
        else:
            result_map[ensembl_id] = ensembl_id
    
    merged_ids = [id for id in unique_ids if is_merged_gene_id(id)]
    for merged_id in merged_ids:
        symbol = query_merged_gene_symbol(merged_id)
        result_map[merged_id] = symbol
    
    save_unmapped_details()
    return result_map

def convert_merged_gene_symbol(ensembl_id, symbol_map):
    if ensembl_id in symbol_map:
        return symbol_map[ensembl_id]
    
    match = re.match(r'__ambiguous\[(.*?)\]', ensembl_id)
    if match:
        content = match.group(1)
        parts = [p for p in content.split('+') if p.strip()]
        converted_parts = []
        for part in parts:
            if part in symbol_map:
                converted_parts.append(symbol_map[part])
            else:
                converted_parts.append(part)
        return '/'.join(converted_parts)
    
    return symbol_map.get(ensembl_id, ensembl_id)

def save_unmapped_genes(tissue_dir, tissue_name):
    if unmapped_genes:
        unmapped_path = os.path.join(tissue_dir, f"{tissue_name}_unmapped_genes_all.txt")
        with open(unmapped_path, 'w', encoding='utf-8') as f:
            f.write("# Unmapped ENSG IDs (all occurrences)\n")
            f.write("# Format: ENSG ID\tCount\n")
            f.write("#" + "="*50 + "\n")
            for gene, count in sorted(unmapped_genes.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{gene}\t{count}\n")
        
    if unmapped_genes_final:
        unmapped_final_path = os.path.join(tissue_dir, f"{tissue_name}_unmapped_genes_final.txt")
        with open(unmapped_final_path, 'w', encoding='utf-8') as f:
            f.write("# Final unmapped ENSG IDs\n")
            f.write("#" + "="*50 + "\n")
            for gene in sorted(unmapped_genes_final):
                f.write(f"{gene}\n")

    save_cache(GENE_SYMBOL_MAP)

# ==================== Helper Functions ====================
def filter_invalid_genes(gene_corr_dict, gene_symbol_map, tissue_dir, tissue_name):
    invalid_genes = []
    valid_genes = []
    
    for gene in gene_corr_dict.keys():
        mapped_symbol = gene_symbol_map.get(gene, gene)
        
        is_invalid = False
        reason = ""
        
        if mapped_symbol and mapped_symbol.strip() == "E":
            is_invalid = True
            reason = "Mapped to a single letter 'E'"
        elif mapped_symbol.startswith('ENSG'):
            is_invalid = True
            reason = "Keep the ENSG ID"
        elif mapped_symbol.startswith('E') and len(mapped_symbol) > 5 and mapped_symbol[1:].replace('.', '').replace('_', '').isdigit():
            is_invalid = True
            reason = "E followed by a number format"
        elif gene.startswith('__ambiguous'):
            parts = parse_merged_gene_id(gene)
            if parts:
                all_parts_invalid = True
                for part in parts:
                    part_symbol = gene_symbol_map.get(part, part)
                    if not (part_symbol.startswith('ENSG') or 
                           (part_symbol.startswith('E') and len(part_symbol) > 5 and part_symbol[1:].replace('.', '').isdigit()) or
                           part_symbol.strip() == "E"):
                        all_parts_invalid = False
                        break
                if all_parts_invalid:
                    is_invalid = True
                    reason = "All parts of the combined gene failed to be mapped successfully"
        
        if is_invalid:
            invalid_genes.append({'gene': gene, 'mapped_symbol': mapped_symbol, 'reason': reason})
        else:
            valid_genes.append(gene)
    
    if invalid_genes:
        invalid_genes_path = os.path.join(tissue_dir, f"{tissue_name}_invalid_genes_details.txt")
        with open(invalid_genes_path, 'w', encoding='utf-8') as f:
            f.write("# Detailed information of invalid genes that failed to be mapped\n")
            f.write("# Format: ENSG\tMapping Result\tInvalid Reason\n")
            f.write("#" + "="*60 + "\n")
            for item in invalid_genes:
                f.write(f"{item['gene']}\t{item['mapped_symbol']}\t{item['reason']}\n")
        
        invalid_ensg_path = os.path.join(tissue_dir, f"{tissue_name}_invalid_genes_list.txt")
        with open(invalid_ensg_path, 'w', encoding='utf-8') as f:
            f.write("# 未The list of invalid genes that were successfully mapped as ENSG\n")
            for item in invalid_genes:
                f.write(f"{item['gene']}\n")
        
        for item in invalid_genes:
            gene_corr_dict.pop(item['gene'], None)
        
        print(f"Filter out {len(invalid_genes)} invalid genes that failed to be mapped successfully")
    
    return valid_genes

def get_display_gene(gene, gene_symbol_map=None):
    if gene_symbol_map is None:
        return gene
    
    if gene in gene_symbol_map:
        if gene.startswith('__ambiguous'):
            return convert_merged_gene_symbol(gene, gene_symbol_map)
        return gene_symbol_map[gene]
    return gene

def save_gene_list(gene_list, output_path, gene_symbol_map=None):
    if not gene_list:
        return
    with open(output_path, 'w', encoding='utf-8') as f:
        for gene in gene_list:
            display_gene = get_display_gene(gene, gene_symbol_map)
            f.write(f"{display_gene}\n")

def check_sample_availability(tissue_samples):    
    valid_samples = []
    invalid_samples = []
    for sample in tissue_samples:
        print(f"\nSamples: {sample}")
        print("-" * 40)
        tissue = get_sample_tissue(sample)
        print(f"Tissue Type: {tissue}")
        ptime_path = os.path.join(PT_DIR, sample, f"{sample}{PSEUDOTIME_FILE_SUFFIX}")
        if os.path.exists(ptime_path):
            try:
                ptime_df = pd.read_csv(ptime_path)
                if PSEUDOTIME_COLUMN in ptime_df.columns:
                    n_valid_ptime = ptime_df[PSEUDOTIME_COLUMN].notna().sum()
                    print(f"Pseudotime: exist ({len(ptime_df)} spots, {n_valid_ptime} vailded {PSEUDOTIME_COLUMN})")
                    ptime_valid = True
                else:
                    print(f"Fake time data: Missing {PSEUDOTIME_COLUMN} column")
                    ptime_valid = False
            except Exception as e:
                print(f"Fake time data: Reading failed {e}")
                ptime_valid = False
        else:
            print(f"Fake time data: File does not exist")
            ptime_valid = False
        
        st_path = os.path.join(ROOT_DIR, sample, "st", f"{sample}.h5ad")
        if os.path.exists(st_path):
            try:
                adata = sc.read_h5ad(st_path)
                print(f"Expression data: There are ({adata.n_obs} spots, {adata.n_vars} genes)")
                
                if ptime_valid:
                    ptime_barcodes = set(ptime_df['barcode'].astype(str))
                    adata_barcodes = set(adata.obs_names.astype(str))
                    common_barcodes = len(ptime_barcodes & adata_barcodes)

                    if common_barcodes >= MIN_SPOTS_PER_SAMPLE:
                        valid_samples.append(sample)
                    else:
                        invalid_samples.append((sample, f"Insufficient total number of spots({common_barcodes})"))
                else:
                    invalid_samples.append((sample, "Invalid pseudotime data"))
            except Exception as e:
                invalid_samples.append((sample, f"Failed to read the data:{e}"))
        else:
            invalid_samples.append((sample, "The data file does not exist."))    
    print(f"\n{'='*60}")
    print(f"Total sample count: {len(tissue_samples)}")
    print(f"Valid sample count: {len(valid_samples)}")
    print(f"Invalid sample count: {len(invalid_samples)}")
    return valid_samples, invalid_samples

def get_gene_expression_stats(tissue_samples):
    gene_sample_count = defaultdict(int)

    for sample in tqdm(tissue_samples, desc="Statistical gene expression"):
        adata = load_gene_expression_data(sample)
        if adata is None:
            continue
        
        sample_genes = set(adata.var_names)
        for gene in sample_genes:
            gene_sample_count[gene] += 1
    
    print(f"The tissue's total sample analysis identified {len(gene_sample_count)} unique genes")
    return gene_sample_count

def filter_genes_by_expression(gene_sample_count, samples, min_fraction=0.5):
    n_samples = len(samples)
    min_samples = max(1, int(n_samples * min_fraction))

    candidate_genes = []
    for gene, count in gene_sample_count.items():
        if count >= min_samples:
            candidate_genes.append((gene, count))
    
    candidate_genes.sort(key=lambda x: x[1], reverse=True)
    print(f"The number of genes that meet more than half of the expression conditions:{len(candidate_genes)}")    
    selected_genes = [gene for gene, _ in candidate_genes]

    return selected_genes

def load_pseudotime_data(sample_name):
    ptime_path = os.path.join(PT_DIR, sample_name, f"{sample_name}{PSEUDOTIME_FILE_SUFFIX}")
    if not os.path.exists(ptime_path):
        return None
    
    try:
        ptime_df = pd.read_csv(ptime_path)
        
        if 'barcode' not in ptime_df.columns or PSEUDOTIME_COLUMN not in ptime_df.columns:
            return None
        
        ptime_df = ptime_df[['barcode', PSEUDOTIME_COLUMN]].copy()
        
        if ptime_df[PSEUDOTIME_COLUMN].isna().all():
            return None
        
        return ptime_df
        
    except Exception:
        return None

def load_gene_expression_data(sample_name):
    st_path = os.path.join(ROOT_DIR, sample_name, "st", f"{sample_name}.h5ad")
    if not os.path.exists(st_path):
        return None
    try:
        adata = sc.read_h5ad(st_path)
        return adata
    except Exception:
        return None

def preprocess_sample_data(sample, selected_genes_list):
    ptime_df = load_pseudotime_data(sample)
    if ptime_df is None:
        print(f"Sample {sample}: The pseudotime data does not exist")
        return None
    
    adata = load_gene_expression_data(sample)
    if adata is None:
        print(f"Sample {sample}: The data being expressed does not exist")
        return None
    
    ptime_barcodes = set(ptime_df['barcode'].astype(str))
    adata_barcodes = set(adata.obs_names.astype(str))
    common_barcodes = list(ptime_barcodes & adata_barcodes)
    
    if len(common_barcodes) < MIN_SPOTS_PER_SAMPLE:
        print(f"{sample}: The number of spots in the sample is less than {MIN_SPOTS_PER_SAMPLE}")
        return None
    
    common_barcodes_sorted = sorted(common_barcodes)
    ptime_common = ptime_df[ptime_df['barcode'].isin(common_barcodes_sorted)].copy()
    ptime_common = ptime_common.sort_values('barcode')
    adata_common = adata[adata.obs_names.isin(common_barcodes_sorted)].copy()
    adata_common = adata_common[common_barcodes_sorted, :]
    
    pseudotime_values = ptime_common[PSEUDOTIME_COLUMN].values.astype(float)
    
    if np.all(np.isnan(pseudotime_values)):
        print(f"{sample}:{PSEUDOTIME_COLUMN} invalid")
        return None
    
    valid_ptime_ratio = np.sum(~np.isnan(pseudotime_values)) / len(pseudotime_values)
    if valid_ptime_ratio < 0.5:
        print(f"{sample}:The proportion of valid {PSEUDOTIME_COLUMN} is too low ({valid_ptime_ratio:.2%})")
        return None
    
    gene_to_idx = {}
    selected_set = set(selected_genes_list)
    
    for i, gene_id in enumerate(adata_common.var_names):
        if gene_id in selected_set:
            gene_to_idx[gene_id] = i
    
    if len(gene_to_idx) == 0:
        print(f"{sample}: None of the genes after screening were expressed. Remove this sample")
        return None
    
    if 'counts' in adata_common.layers:
        X = adata_common.layers['counts']
    else:
        X = adata_common.X
    
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    gene_indices = list(gene_to_idx.values())
    if gene_indices:
        X_filtered = X_dense[:, gene_indices]
        non_zero_mask = (~np.isnan(X_filtered)) & (X_filtered != 0)
        non_zero_count = np.sum(non_zero_mask)
        total_elements = X_filtered.size
        non_zero_ratio = non_zero_count / total_elements if total_elements > 0 else 0
        
        if non_zero_ratio < MIN_EXPRESSION_RATIO:
            print(f"{sample}: The proportion of effective expressions is too low ({non_zero_ratio:.4f} < {MIN_EXPRESSION_RATIO})")
            return None
    
    return {
        'pseudotime': pseudotime_values,
        'gene_to_idx': gene_to_idx,
        'X_dense': X_dense,
        'n_spots': len(common_barcodes),
        'barcodes': common_barcodes_sorted
    }

def compute_correlations_batch(sample_data_dict, gene_batch, sample_list):
    batch_results = {gene: {} for gene in gene_batch}
    
    for sample in sample_list:
        if sample not in sample_data_dict:
            for gene in gene_batch:
                batch_results[gene][sample] = np.nan
            continue
        
        data = sample_data_dict[sample]
        pseudotime = data['pseudotime']
        gene_to_idx = data['gene_to_idx']
        X_dense = data['X_dense']
        
        for gene in gene_batch:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                expression = X_dense[:, gene_idx]
                
                valid_mask = np.isfinite(expression) & np.isfinite(pseudotime)
                if valid_mask.sum() >= 5:
                    corr, _ = spearmanr(expression[valid_mask], pseudotime[valid_mask])
                    batch_results[gene][sample] = corr
                else:
                    batch_results[gene][sample] = np.nan
            else:
                batch_results[gene][sample] = np.nan
    
    return batch_results

def compute_gene_correlations_for_tissue(tissue_name, tissue_samples, selected_genes_list):
    print(f"\nCalculating the correlation of genes in {tissue_name} (using {PSEUDOTIME_NAME} pseudotime)...\n")
    print(f"  Number of genes: {len(selected_genes_list)}")
    print(f"  Number of samples: {len(tissue_samples)}")
    print(f"  Batch size: {BATCH_SIZE} genes per batch")
    
    sample_data_dict = {}
    sample_info_dict = {}
    sample_raw_data_dict = {}
    
    valid_count = 0
    for sample in tqdm(tissue_samples, desc="Preprocessed samples"):
        data = preprocess_sample_data(sample, selected_genes_list)
        if data is not None:
            sample_data_dict[sample] = data
            sample_raw_data_dict[sample] = data
            sample_info_dict[sample] = {'n_spots': data['n_spots']}
            valid_count += 1
    
    valid_samples = list(sample_data_dict.keys())
    print(f"Number of effective samples after preprocessing:{valid_count}/{len(tissue_samples)}")
    
    deleted_samples = len(tissue_samples) - valid_count
    if deleted_samples > 0:
        print(f" {deleted_samples} samples that showed no expression in the filtered genes have been deleted")
    
    if not valid_samples:
        print(f"No valid samples available, therefore terminating the analysis")
        return None, None, None, None
    
    gene_corr_dict = {}
    all_results = []
    
    n_batches = (len(selected_genes_list) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nStart calculating the correlations in batches (with {n_batches} batches)...")
    
    for batch_idx in tqdm(range(n_batches), desc="Processing batch"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(selected_genes_list))
        gene_batch = selected_genes_list[start_idx:end_idx]
        
        batch_results = compute_correlations_batch(sample_data_dict, gene_batch, valid_samples)
        
        for gene, sample_corrs in batch_results.items():
            gene_corr_dict[gene] = sample_corrs
            for sample, corr in sample_corrs.items():
                if not np.isnan(corr):
                    all_results.append({
                        'tissue': tissue_name,
                        'sample': sample,
                        'gene': gene,
                        'correlation': corr,
                        'n_spots': sample_info_dict[sample]['n_spots']
                    })
        
        gc.collect()

    print("\nProcessing completed:")
    print(f"Number of valid genes:  {len(gene_corr_dict)}")
    
    all_results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return gene_corr_dict, sample_info_dict, all_results_df, sample_raw_data_dict

# ==================== Filter function ====================
def select_top_positive_genes_by_count_adaptive(gene_corr_dict, selected_genes, samples, n_top=50, require_half_samples=True):
    """
    Adaptive selection of positively correlated genes: Start from 0.5 and decrease the threshold by 0.01 increments until n_top genes are found.
    Return the list of positively correlated genes sorted in descending order by sample count.
    When the sample counts are the same, sort by average correlation from high to low.
    Use the >= threshold criterion.
    """
    current_threshold = 0.5
    min_threshold = 0.01
    step = 0.01
    
    print(f"\nFilter out positively correlated genes (based on count, target: {n_top} genes)...")
    
    best_selected = []
    best_threshold = current_threshold
    
    while current_threshold >= min_threshold - 1e-10:
        print(f"\n Threshold for attempt: r ≥ {current_threshold:.2f}")
        
        gene_stats = []
        n_samples = len(samples)
        min_required_samples = int(n_samples * 0.5) if require_half_samples else 0
        
        for gene in tqdm(selected_genes, desc=f"Screening of positively correlated genes ( Threshold {current_threshold:.2f})", leave=False):
            corrs = [gene_corr_dict[gene].get(s, np.nan) for s in samples]
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            
            if require_half_samples and len(valid_corrs) < min_required_samples:
                continue
                
            high_corr_count = sum(1 for c in valid_corrs if c >= current_threshold)
            if high_corr_count > 0:
                avg_corr = np.mean(valid_corrs)
                gene_stats.append((gene, high_corr_count, avg_corr, len(valid_corrs)))
        
        if not gene_stats:
            print(f"No positive correlation genes were found")
            current_threshold -= step
            continue
        
        gene_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        selected = gene_stats[:min(n_top, len(gene_stats))]
        found_count = len(selected)
        
        print(f"Found {found_count} positively correlated genes")

        best_selected = selected
        best_threshold = current_threshold
        
        if found_count >= n_top:
            print(f"\nFound {found_count} positively correlated genes (threshold {current_threshold:.2f})")
            break
        else:
            print(f"Only {found_count} were found. Continue to lower the threshold...")
            current_threshold -= step
    
    if not best_selected:
        print(f"\nNo positive correlation genes were found at any of the tested thresholds")
        return [], None, None, best_threshold
    
    selected_genes_list = [g[0] for g in best_selected]
    sort_key_dict = {gene: count for gene, count, avg, n_valid in best_selected}
    selected_stats = pd.DataFrame(best_selected, columns=['gene', 'n_samples_r_ge_threshold', 'avg_correlation', 'n_valid_samples'])
    
    for i, (gene, count, avg, n_valid) in enumerate(best_selected):
        tie_info = "(Parallel)" if i > 0 and best_selected[i-1][1] == count else ""
        print(f"    {i+1:2d}. {gene:20s} | {count:3d} samples r≥{best_threshold:.2f}{tie_info} | avg r={avg:.3f} | valid={n_valid}")
    
    return selected_genes_list, selected_stats, sort_key_dict, best_threshold

def select_top_negative_genes_by_count_adaptive(gene_corr_dict, selected_genes, samples, n_top=50, require_half_samples=True):
    """
    Adaptive selection of negatively correlated genes: Start from 0.5 and decrease the threshold by 0.01 increments until n_top genes are found.
    Return the list of negatively correlated genes sorted in descending order by the number of samples.
    When the number of samples is the same, sort them in ascending order by the average correlation.
    Use <= -threshold for judgment.
    """
    current_threshold = 0.5
    min_threshold = 0.01
    step = 0.01
    
    print(f"\nAdaptive filtering of negatively correlated genes (based on count, objective: {n_top} genes)...")
    
    best_selected = []
    best_threshold = current_threshold
    
    while current_threshold >= min_threshold - 1e-10:
        print(f"\nThreshold for attempt: r ≤ -{current_threshold:.2f}")
        
        gene_stats = []
        n_samples = len(samples)
        min_required_samples = int(n_samples * 0.5) if require_half_samples else 0
        
        for gene in tqdm(selected_genes, desc=f"Screening of negatively correlated genes (threshold {current_threshold:.2f})", leave=False):
            corrs = [gene_corr_dict[gene].get(s, np.nan) for s in samples]
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            
            if require_half_samples and len(valid_corrs) < min_required_samples:
                continue
                
            low_corr_count = sum(1 for c in valid_corrs if c <= -current_threshold)
            if low_corr_count > 0:
                avg_corr = np.mean(valid_corrs)
                gene_stats.append((gene, low_corr_count, avg_corr, len(valid_corrs)))
        
        if not gene_stats:
            print(f"No negative correlation genes were found")
            current_threshold -= step
            continue
        
        gene_stats.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        selected = gene_stats[:min(n_top, len(gene_stats))]
        found_count = len(selected)
        
        print(f"Found {found_count} negatively correlated genes")
        
        best_selected = selected
        best_threshold = current_threshold
        
        if found_count >= n_top:
            print(f"\nFound {found_count} negatively correlated genes (threshold {current_threshold:.2f})")
            break
        else:
            print(f"Only {found_count} were found. Continue to lower the threshold...")
            current_threshold -= step
    
    if not best_selected:
        print(f"\nNo negative correlation genes were found at any of the tested thresholds")
        return [], None, None, best_threshold
    
    selected_genes_list = [g[0] for g in best_selected]
    sort_key_dict = {gene: count for gene, count, avg, n_valid in best_selected}
    selected_stats = pd.DataFrame(best_selected, columns=['gene', 'n_samples_r_le_neg_threshold', 'avg_correlation', 'n_valid_samples'])

    for i, (gene, count, avg, n_valid) in enumerate(best_selected):
        tie_info = "(Parallel)" if i > 0 and best_selected[i-1][1] == count else ""
        print(f"    {i+1:2d}. {gene:20s} | {count:3d} samples r≤-{best_threshold:.2f}{tie_info} | avg r={avg:.3f} | valid={n_valid}")
    
    return selected_genes_list, selected_stats, sort_key_dict, best_threshold

def select_top_positive_genes_by_avg(gene_corr_dict, selected_genes, samples, n_top=50, require_half_samples=True):
    gene_stats = []
    n_samples = len(samples)
    min_required_samples = int(n_samples * 0.5) if require_half_samples else 0
    
    print(f"\nFilter out positively correlated genes (sorted by avg, and select the top {n_top} ones)...")
    
    for gene in tqdm(selected_genes, desc="Screening of positively correlated genes"):
        corrs = [gene_corr_dict[gene].get(s, np.nan) for s in samples]
        valid_corrs = [c for c in corrs if not np.isnan(c)]
        
        if require_half_samples and len(valid_corrs) < min_required_samples:
            continue
            
        if valid_corrs:
            avg_corr = np.mean(valid_corrs)
            pos_corr_count = sum(1 for c in valid_corrs if c > 0)
            gene_stats.append((gene, avg_corr, pos_corr_count, len(valid_corrs)))
    
    if not gene_stats:
        print(f"No positive correlation genes were found")
        return [], None, None
    
    gene_stats.sort(key=lambda x: x[1], reverse=True)
    selected = gene_stats[:min(n_top, len(gene_stats))]
    
    selected_genes_list = [g[0] for g in selected]
    sort_key_dict = {gene: avg for gene, avg, pos_count, n_valid in selected}
    selected_stats = pd.DataFrame(selected, columns=['gene', 'avg_correlation', 'n_positive_samples', 'n_valid_samples'])

    for i, (gene, avg, pos_count, n_valid) in enumerate(selected):
        tie_info = "(Parallel)" if i > 0 and abs(selected[i-1][1] - avg) < 1e-10 else ""
        print(f"    {i+1:2d}. {gene:20s} | avg r={avg:.3f}{tie_info} | pos samples={pos_count:3d} | valid={n_valid}")
    
    return selected_genes_list, selected_stats, sort_key_dict

def select_top_negative_genes_by_avg(gene_corr_dict, selected_genes, samples, n_top=50, require_half_samples=True):
    gene_stats = []
    n_samples = len(samples)
    min_required_samples = int(n_samples * 0.5) if require_half_samples else 0
    
    print(f"\nFilter out negatively correlated genes (sorted by avg, and select the top {n_top} ones)...")
    
    for gene in tqdm(selected_genes, desc="Screen for negatively correlated genes"):
        corrs = [gene_corr_dict[gene].get(s, np.nan) for s in samples]
        valid_corrs = [c for c in corrs if not np.isnan(c)]
        
        if require_half_samples and len(valid_corrs) < min_required_samples:
            continue
            
        if valid_corrs:
            avg_corr = np.mean(valid_corrs)
            neg_corr_count = sum(1 for c in valid_corrs if c < 0)
            gene_stats.append((gene, avg_corr, neg_corr_count, len(valid_corrs)))
    
    if not gene_stats:
        print(f"No negative correlation genes were found")
        return [], None, None
    
    gene_stats.sort(key=lambda x: x[1])
    selected = gene_stats[:min(n_top, len(gene_stats))]
    
    selected_genes_list = [g[0] for g in selected]
    sort_key_dict = {gene: avg for gene, avg, neg_count, n_valid in selected}
    selected_stats = pd.DataFrame(selected, columns=['gene', 'avg_correlation', 'n_negative_samples', 'n_valid_samples'])
    
    for i, (gene, avg, neg_count, n_valid) in enumerate(selected):
        tie_info = "(Parallel)" if i > 0 and abs(selected[i-1][1] - avg) < 1e-10 else ""
        print(f"    {i+1:2d}. {gene:20s} | avg r={avg:.3f}{tie_info} | neg samples={neg_count:3d} | valid={n_valid}")
    
    return selected_genes_list, selected_stats, sort_key_dict
# ==================== Heatmap plotting function ====================
def calculate_optimal_figure_size(n_genes, n_samples, max_pixels=MAX_PIXELS, base_dpi=300, row_spacing=1.5):
    """
    Calculate the optimal graphic size
    Adjust the row height and column width dynamically according to the number of genes
    """
    BASE_INCHES_PER_GENE = 0.28
    
    if n_genes <= 20:
        base_per_gene = BASE_INCHES_PER_GENE
    elif n_genes <= 30:
        base_per_gene = BASE_INCHES_PER_GENE * 0.95
    elif n_genes <= 50:
        base_per_gene = BASE_INCHES_PER_GENE * 0.95
    elif n_genes <= 100:
        base_per_gene = BASE_INCHES_PER_GENE * 0.85
    elif n_genes <= 200:
        base_per_gene = BASE_INCHES_PER_GENE * 0.80
    elif n_genes <= 300:
        base_per_gene = BASE_INCHES_PER_GENE * 0.75
    else:
        base_per_gene = BASE_INCHES_PER_GENE * 0.70
    
    INCHES_PER_GENE = base_per_gene * row_spacing
    
    if n_samples == 1:
        INCHES_PER_SAMPLE = 0.8
    elif n_samples <= 5:
        INCHES_PER_SAMPLE = 2.5
    elif n_samples <= 10:
        INCHES_PER_SAMPLE = 2
    elif n_samples <= 20:
        INCHES_PER_SAMPLE = 1.5
    elif n_samples <= 30:
        INCHES_PER_SAMPLE = 0.6
    elif n_samples <= 50:
        INCHES_PER_SAMPLE = 0.5
    elif n_samples <= 100:
        INCHES_PER_SAMPLE = 0.4
    else:
        INCHES_PER_SAMPLE = 0.3
    
    base_height = n_genes * INCHES_PER_GENE
    base_width = n_samples * INCHES_PER_SAMPLE
    
    fig_height = base_height + 4.0
    if n_samples <= 5:
        fig_width = base_width + 2.0
    else:
        fig_width = base_width + 6.0
    
    height_pixels = fig_height * base_dpi
    width_pixels = fig_width * base_dpi
    
    if height_pixels > max_pixels or width_pixels > max_pixels:
        needed_dpi_height = max_pixels / fig_height
        needed_dpi_width = max_pixels / fig_width
        dpi = min(needed_dpi_height, needed_dpi_width, base_dpi)
    else:
        dpi = base_dpi
    
    if n_genes < 20:
        gene_font_size = 36
        value_font_size = 36
    elif n_genes < 30:
        gene_font_size = 36
        value_font_size = 36
    elif n_genes < 50:
        gene_font_size = 36
        value_font_size = 36
    elif n_genes < 100:
        gene_font_size = 28
        value_font_size = 28
    elif n_genes < 200:
        gene_font_size = 28
        value_font_size = 28
    elif n_genes < 300:
        gene_font_size = 28
        value_font_size = 28
    elif n_genes < 500:
        gene_font_size = 24
        value_font_size = 24
    elif n_genes < 1000:
        gene_font_size = 20
        value_font_size = 20
    elif n_genes < 2000:
        gene_font_size = 20
        value_font_size = 20
    else:
        gene_font_size = 16
        value_font_size = 16

    return fig_width, fig_height, dpi, gene_font_size, value_font_size

def plot_correlation_heatmap(gene_corr_dict, selected_genes, samples, tissue_name, output_dir, 
                             title_suffix, filename_prefix, sort_key_dict=None, 
                             gene_symbol_map=None, sort_ascending=False, row_spacing=1.5):
    if not selected_genes:
        print(f"There are no genes that can be drawn")
        return False

    display_genes = []
    if gene_symbol_map:
        for gene in selected_genes:
            if gene in gene_symbol_map:
                if gene.startswith('__ambiguous'):
                    display_genes.append(convert_merged_gene_symbol(gene, gene_symbol_map))
                else:
                    display_genes.append(gene_symbol_map[gene])
            else:
                display_genes.append(gene)
    else:
        display_genes = selected_genes
    
    corr_matrix = np.full((len(selected_genes), len(samples)), np.nan)
    for i, gene in enumerate(selected_genes):
        for j, sample in enumerate(samples):
            corr_matrix[i, j] = gene_corr_dict[gene].get(sample, np.nan)
    
    corr_df = pd.DataFrame(corr_matrix, index=display_genes, columns=samples)
    
    if sort_key_dict is not None:
        display_sort_key = {}
        for i, gene in enumerate(selected_genes):
            display_gene = display_genes[i]
            if gene in sort_key_dict:
                display_sort_key[display_gene] = sort_key_dict[gene]
        
        if sort_ascending:
            gene_names_sorted = sorted(display_genes, key=lambda x: display_sort_key.get(x, 0), reverse=False)
            print(f"Sort in ascending order by the sorting key (the smallest value is at the top)")
        else:
            gene_names_sorted = sorted(display_genes, key=lambda x: display_sort_key.get(x, 0), reverse=True)
            print(f"Sort in descending order by the sorting key (the largest value is at the top)")
        
    else:
        gene_names_sorted = display_genes

    sample_names_sorted = samples
    corr_df_sorted = corr_df.loc[gene_names_sorted, sample_names_sorted]
    
    fig_width, fig_height, dpi, gene_font_size, value_font_size = calculate_optimal_figure_size(
        len(gene_names_sorted), len(sample_names_sorted), row_spacing=row_spacing
    )
    
    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(corr_df_sorted.values, aspect='auto', cmap=COLORMAP,
                       vmin=-1, vmax=1, interpolation='nearest')
        
        ax.set_yticks(range(len(gene_names_sorted)))
        ax.set_yticklabels(gene_names_sorted, fontsize=gene_font_size)
        ax.set_xticks(range(len(sample_names_sorted)))
        ax.set_xticklabels(sample_names_sorted, rotation=90, fontsize=gene_font_size)
        ax.set_xlabel('Samples', fontsize=36)
        ax.set_ylabel('Genes', fontsize=36)    
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman Correlation', fontsize=36)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=36)
        ax.set_title(f'{tissue_name} - {title_suffix}', fontsize=36, pad=12)
        plt.tight_layout()
        
        pdf_path = os.path.join(output_dir, f"{filename_prefix}_heatmap.pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
        png_path = os.path.join(output_dir, f"{filename_prefix}_heatmap.png")
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        
        data_path = os.path.join(output_dir, f"{filename_prefix}_data.csv")
        corr_df_sorted.to_csv(data_path, encoding='utf-8-sig')
        
        return True
        
    except Exception as e:
        return False

def plot_combined_heatmap(gene_corr_dict, positive_genes, negative_genes, samples, tissue_name, output_dir, n_top, sort_by='count', gene_symbol_map=None, row_spacing=1.5):
    if not positive_genes and not negative_genes:
        print(f"There are no genes that can be drawn")
        return False
    
    combined_genes = []
    gene_types = []
    
    if positive_genes:
        combined_genes.extend(positive_genes)
        gene_types.extend(['Positive'] * len(positive_genes))
    if negative_genes:
        combined_genes.extend(negative_genes)
        gene_types.extend(['Negative'] * len(negative_genes))

    display_genes = []
    if gene_symbol_map:
        for gene in combined_genes:
            if gene in gene_symbol_map:
                if gene.startswith('__ambiguous'):
                    display_genes.append(convert_merged_gene_symbol(gene, gene_symbol_map))
                else:
                    display_genes.append(gene_symbol_map[gene])
            else:
                display_genes.append(gene)
    else:
        display_genes = combined_genes
    
    sort_display = "According to the sample count" if sort_by == 'count' else "According to average correlation"

    n_genes = len(combined_genes)
    n_samples = len(samples)
    
    corr_matrix = np.full((n_genes, n_samples), np.nan)
    for i, gene in enumerate(combined_genes):
        for j, sample in enumerate(samples):
            corr_matrix[i, j] = gene_corr_dict[gene].get(sample, np.nan)
    
    corr_df = pd.DataFrame(corr_matrix, index=display_genes, columns=samples)
    
    genes_sorted = display_genes
    samples_sorted = samples
    corr_df_sorted = corr_df.loc[genes_sorted, samples_sorted]

    fig_width, fig_height, dpi, gene_font_size, value_font_size = calculate_optimal_figure_size(
        len(genes_sorted), len(samples_sorted), row_spacing=row_spacing
    )
    
    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(corr_df_sorted.values, aspect='auto', cmap=COLORMAP,
                       vmin=-1, vmax=1, interpolation='nearest')
        
        ax.set_yticks(range(len(genes_sorted)))
        ax.set_yticklabels(genes_sorted, fontsize=gene_font_size)
        ax.set_xticks(range(len(samples_sorted)))
        ax.set_xticklabels(samples_sorted, rotation=90, fontsize=gene_font_size)
        ax.set_xlabel('Samples', fontsize=36)
        ax.set_ylabel('Genes', fontsize=36)        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman Correlation', fontsize=36)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=36)
        ax.set_title(f'{tissue_name} - Combined Top {n_top} (Positive + Negative, both {sort_display})', 
                     fontsize=36, pad=15)
        plt.tight_layout()
        
        pdf_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_heatmap_{sort_by}.pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
        png_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_heatmap_{sort_by}.png")
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        
        data_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_data_{sort_by}.csv")
        corr_df_sorted.to_csv(data_path, encoding='utf-8-sig')
        
        print(f"The merged heatmap has been saved")
        print(f"file: {os.path.basename(pdf_path)}")
        
        type_df = pd.DataFrame({
            'gene': combined_genes,
            'display_name': display_genes,
            'type': gene_types
        })
        type_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_gene_types_{sort_by}.csv")
        type_df.to_csv(type_path, index=False, encoding='utf-8-sig')
        
        return True
        
    except Exception as e:
        print(f"Failed to draw the merged heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_combined_positive_by_count_negative_by_avg_heatmap(gene_corr_dict, positive_genes, negative_by_avg_genes, 
                                                            samples, tissue_name, output_dir, n_top, 
                                                            gene_symbol_map=None, row_spacing=1.5):
    if not positive_genes and not negative_by_avg_genes:
        print(f"There are no genes that can be drawn")
        return False
    
    positive_genes_copy = positive_genes.copy() if positive_genes else []
    negative_genes_copy = negative_by_avg_genes.copy() if negative_by_avg_genes else []
    positive_sorted = positive_genes_copy

    neg_gene_avgs = {}
    for gene in negative_genes_copy:
        corrs = [gene_corr_dict[gene].get(s, np.nan) for s in samples]
        valid_corrs = [c for c in corrs if not np.isnan(c)]
        if valid_corrs:
            neg_gene_avgs[gene] = np.mean(valid_corrs)
        else:
            neg_gene_avgs[gene] = 0
    
    negative_sorted = sorted(negative_genes_copy, key=lambda x: neg_gene_avgs.get(x, 0))
    combined_genes = positive_sorted + negative_sorted
    gene_types = ['Positive'] * len(positive_sorted) + ['Negative'] * len(negative_sorted)

    display_genes = []
    if gene_symbol_map:
        for gene in combined_genes:
            if gene in gene_symbol_map:
                if gene.startswith('__ambiguous'):
                    display_genes.append(convert_merged_gene_symbol(gene, gene_symbol_map))
                else:
                    display_genes.append(gene_symbol_map[gene])
            else:
                display_genes.append(gene)
    else:
        display_genes = combined_genes
    n_genes = len(combined_genes)
    n_samples = len(samples)

    corr_matrix = np.full((n_genes, n_samples), np.nan)
    for i, gene in enumerate(combined_genes):
        for j, sample in enumerate(samples):
            corr_matrix[i, j] = gene_corr_dict[gene].get(sample, np.nan)
    
    corr_df = pd.DataFrame(corr_matrix, index=display_genes, columns=samples)
    
    genes_sorted = display_genes

    samples_sorted = samples
    
    corr_df_sorted = corr_df.loc[genes_sorted, samples_sorted]
    
    fig_width, fig_height, dpi, gene_font_size, value_font_size = calculate_optimal_figure_size(
        len(genes_sorted), len(samples_sorted), row_spacing=row_spacing
    )
    
    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(corr_df_sorted.values, aspect='auto', cmap=COLORMAP,
                       vmin=-1, vmax=1, interpolation='nearest')
        
        ax.set_yticks(range(len(genes_sorted)))
        ax.set_yticklabels(genes_sorted, fontsize=gene_font_size)
        ax.set_xticks(range(len(samples_sorted)))
        ax.set_xticklabels(samples_sorted, rotation=90, fontsize=gene_font_size)
        ax.set_xlabel('Samples', fontsize=36)
        ax.set_ylabel('Genes', fontsize=36)    
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman Correlation', fontsize=36)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=36)
        ax.set_title(f'{tissue_name} - Combined Top {n_top} (Positive by count + Negative by avg)', 
                     fontsize=36, pad=15)
        plt.tight_layout()
        
        pdf_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_pos_count_neg_avg.pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
        png_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_pos_count_neg_avg.png")
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        
        data_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_pos_count_neg_avg_data.csv")
        corr_df_sorted.to_csv(data_path, encoding='utf-8-sig')
        
        print(f"The merged heatmap has been saved")
        print(f"file: {os.path.basename(pdf_path)}")
        
        type_df = pd.DataFrame({
            'gene': combined_genes,
            'display_name': display_genes,
            'type': gene_types,
            'avg_correlation': [neg_gene_avgs.get(g, 0) if g in negative_genes_copy else None for g in combined_genes]
        })
        type_path = os.path.join(output_dir, f"{tissue_name}_top{n_top}_combined_gene_types.csv")
        type_df.to_csv(type_path, index=False, encoding='utf-8-sig')
        
        print(f"\nThe sequence of negatively correlated genes in the final heat map:")
        neg_start_idx = len(positive_sorted)
        for i, gene in enumerate(genes_sorted[neg_start_idx:]):
            original_gene = combined_genes[neg_start_idx + i]
            print(f"    {i+1:2d}. {gene} (origin: {original_gene})")
        
        return True
        
    except Exception as e:
        return False
# ==================== Main Analysis Functions ====================
def analyze_tissue(tissue_name, tissue_samples):
    print(f"Analysis organization: {tissue_name}")
    print(f"Number of samples: {len(tissue_samples)}")
    print(f"Pseudotime type: {PSEUDOTIME_NAME}")
    
    need_mapping = tissue_name not in NO_MAPPING_TISSUES
    print(f"Genetic mapping: {'Required' if need_mapping else 'Not Required'}")
    
    global unmapped_genes, unmapped_genes_final
    unmapped_genes.clear()
    unmapped_genes_final.clear()
    
    valid_samples, invalid_samples = check_sample_availability(tissue_samples)
    if not valid_samples:
        print(f"No valid samples")
        return False
    
    start_time = time.time()
    tissue_dir = os.path.join(OUTPUT_DIR, tissue_name)
    os.makedirs(tissue_dir, exist_ok=True)
    
    gene_sample_count = get_gene_expression_stats(valid_samples)
    selected_genes_list = filter_genes_by_expression(
        gene_sample_count, valid_samples, MIN_SAMPLE_FRACTION
    )
    
    if not selected_genes_list:
        print(f"No genes were selected through the screening process")
        return False
    
    selected_genes_df = pd.DataFrame({'gene': selected_genes_list})
    selected_genes_path = os.path.join(tissue_dir, f"{tissue_name}_selected_genes.csv.gz")
    selected_genes_df.to_csv(selected_genes_path, index=False, compression='gzip', encoding='utf-8-sig')
    
    gene_corr_dict, sample_info_dict, all_results_df, sample_raw_data_dict = compute_gene_correlations_for_tissue(
        tissue_name, valid_samples, selected_genes_list
    )
    if not gene_corr_dict:
        print(f"No valid data")
        return False
    
    samples = list(sample_info_dict.keys()) if sample_info_dict else []
    if not samples:
        print(f"No valid samples")
        return False
    
    print(f"\nFinal analysis sample size: {len(samples)}")
    print(f"Final analysis of the number of genes:{len(gene_corr_dict)}")
    
    gene_symbol_map = None
    if need_mapping:
        print(f"\nBatch query of gene symbols is in progress...")
        all_genes = list(gene_corr_dict.keys())
        gene_symbol_map = batch_convert_to_symbols(all_genes, need_mapping=True)
        mapped_count = sum(1 for g in all_genes if gene_symbol_map.get(g, g) != g)
        print(f"Successfully mapped {mapped_count} / {len(all_genes)} genes")
        
        filter_invalid_genes(gene_corr_dict, gene_symbol_map, tissue_dir, tissue_name)
        print(f"Number of effective genes after filtration:{len(gene_corr_dict)}")
        
        selected_genes_list = [g for g in selected_genes_list if g in gene_corr_dict]
        save_unmapped_genes(tissue_dir, tissue_name)
    
    if not all_results_df.empty:
        if need_mapping and gene_symbol_map is not None:
            all_results_df['gene_symbol'] = all_results_df['gene'].map(lambda x: gene_symbol_map.get(x, x))
            for i, row in all_results_df.iterrows():
                if row['gene'].startswith('__ambiguous'):
                    all_results_df.loc[i, 'gene_symbol'] = convert_merged_gene_symbol(row['gene'], gene_symbol_map)
        else:
            all_results_df['gene_symbol'] = all_results_df['gene']
        
        ptime_suffix = f"_{PSEUDOTIME_TYPE}"
        results_path = os.path.join(tissue_dir, f"{tissue_name}_all_correlations{ptime_suffix}.csv.gz")
        all_results_df.to_csv(results_path, index=False, compression='gzip', encoding='utf-8-sig')
    
    if not gene_corr_dict:
        print(f" No valid genes were found")
        return False
    
    top_genes_dict = {}
    for n_top in TOP_N_GENES_LIST:
        print(f"\n{'='*50}")
        print(f"Process Top {n_top} Gene")
        print(f"{'='*50}")
        
        positive_by_count, positive_stats_count, positive_sort_key_count, pos_threshold = select_top_positive_genes_by_count_adaptive(
            gene_corr_dict, selected_genes_list, samples, n_top=n_top, require_half_samples=True
        )
        positive_by_avg, positive_stats_avg, positive_sort_key_avg = select_top_positive_genes_by_avg(
            gene_corr_dict, selected_genes_list, samples, n_top=n_top, require_half_samples=True
        )
        negative_by_count, negative_stats_count, negative_sort_key_count, neg_count_threshold = select_top_negative_genes_by_count_adaptive(
            gene_corr_dict, selected_genes_list, samples, n_top=n_top, require_half_samples=True
        )
        negative_by_avg, negative_stats_avg, negative_sort_key_avg = select_top_negative_genes_by_avg(
            gene_corr_dict, selected_genes_list, samples, n_top=n_top, require_half_samples=True
        )
        
        top_genes_dict[n_top] = {
            'positive_by_count': positive_by_count if positive_by_count else [],
            'positive_by_avg': positive_by_avg if positive_by_avg else [],
            'negative_by_count': negative_by_count if negative_by_count else [],
            'negative_by_avg': negative_by_avg if negative_by_avg else []
        }
        
        topn_dir = os.path.join(tissue_dir, f"top{n_top}_results_{PSEUDOTIME_TYPE}")
        os.makedirs(topn_dir, exist_ok=True)
        

        if positive_by_count:
            pos_list_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_positive_by_count_gene_names.txt")
            save_gene_list(positive_by_count, pos_list_path, gene_symbol_map if need_mapping else None)
            
            if positive_stats_count is not None:
                if need_mapping and gene_symbol_map is not None:
                    positive_stats_count['gene_symbol'] = positive_stats_count['gene'].map(lambda x: gene_symbol_map.get(x, x))
                pos_stats_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_positive_by_count_stats.csv")
                positive_stats_count.to_csv(pos_stats_path, index=False, encoding='utf-8-sig')
            
            plot_correlation_heatmap(
                gene_corr_dict, positive_by_count, samples, tissue_name, topn_dir,
                title_suffix=f'Top {n_top} Positive (by count, thr={pos_threshold:.2f})',
                filename_prefix=f"{tissue_name}_top{n_top}_positive_by_count",
                sort_key_dict=positive_sort_key_count,
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                sort_ascending=False, row_spacing=1.5
            )
                    
  
        if positive_by_avg:
            pos_avg_list_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_positive_by_avg_gene_names.txt")
            save_gene_list(positive_by_avg, pos_avg_list_path, gene_symbol_map if need_mapping else None)
            
            if positive_stats_avg is not None:
                if need_mapping and gene_symbol_map is not None:
                    positive_stats_avg['gene_symbol'] = positive_stats_avg['gene'].map(lambda x: gene_symbol_map.get(x, x))
                pos_stats_avg_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_positive_by_avg_stats.csv")
                positive_stats_avg.to_csv(pos_stats_avg_path, index=False, encoding='utf-8-sig')
            
            plot_correlation_heatmap(
                gene_corr_dict, positive_by_avg, samples, tissue_name, topn_dir,
                title_suffix=f'Top {n_top} Positive (by avg)',
                filename_prefix=f"{tissue_name}_top{n_top}_positive_by_avg",
                sort_key_dict=positive_sort_key_avg,
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                sort_ascending=False, row_spacing=1.5
            )
            
        
        
        if negative_by_count:
            neg_count_list_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_negative_by_count_gene_names.txt")
            save_gene_list(negative_by_count, neg_count_list_path, gene_symbol_map if need_mapping else None)
            
            if negative_stats_count is not None:
                if need_mapping and gene_symbol_map is not None:
                    negative_stats_count['gene_symbol'] = negative_stats_count['gene'].map(lambda x: gene_symbol_map.get(x, x))
                neg_stats_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_negative_by_count_stats.csv")
                negative_stats_count.to_csv(neg_stats_path, index=False, encoding='utf-8-sig')
            
            plot_correlation_heatmap(
                gene_corr_dict, negative_by_count, samples, tissue_name, topn_dir,
                title_suffix=f'Top {n_top} Negative (by count, thr={neg_count_threshold:.2f})',
                filename_prefix=f"{tissue_name}_top{n_top}_negative_by_count",
                sort_key_dict=negative_sort_key_count,
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                sort_ascending=False, row_spacing=1.5
            )
            
        
        if negative_by_avg:
            neg_avg_list_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_negative_by_avg_gene_names.txt")
            save_gene_list(negative_by_avg, neg_avg_list_path, gene_symbol_map if need_mapping else None)
            
            if negative_stats_avg is not None:
                if need_mapping and gene_symbol_map is not None:
                    negative_stats_avg['gene_symbol'] = negative_stats_avg['gene'].map(lambda x: gene_symbol_map.get(x, x))
                neg_stats_avg_path = os.path.join(topn_dir, f"{tissue_name}_top{n_top}_negative_by_avg_stats.csv")
                negative_stats_avg.to_csv(neg_stats_avg_path, index=False, encoding='utf-8-sig')
            
            plot_correlation_heatmap(
                gene_corr_dict, negative_by_avg, samples, tissue_name, topn_dir,
                title_suffix=f'Top {n_top} Negative (by avg)',
                filename_prefix=f"{tissue_name}_top{n_top}_negative_by_avg",
                sort_key_dict=negative_sort_key_avg,
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                sort_ascending=True, row_spacing=1.5
            )
            
    
    print(f"\n{'='*50}")
    print("Draw a merged heatmap across the top N items")
    print(f"{'='*50}")
    
    combined_dir = os.path.join(tissue_dir, f"combined_heatmaps_{PSEUDOTIME_TYPE}")
    os.makedirs(combined_dir, exist_ok=True)
    
    for n_top in TOP_N_GENES_LIST:
        pos_count = top_genes_dict[n_top].get('positive_by_count', [])
        pos_avg = top_genes_dict[n_top].get('positive_by_avg', [])
        neg_count = top_genes_dict[n_top].get('negative_by_count', [])
        neg_avg = top_genes_dict[n_top].get('negative_by_avg', [])
        
        if pos_count and neg_count:
            plot_combined_heatmap(
                gene_corr_dict, pos_count, neg_count, samples,
                tissue_name, combined_dir, n_top, sort_by='count',
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                row_spacing=1.5
            )
        
        if pos_avg and neg_avg:
            plot_combined_heatmap(
                gene_corr_dict, pos_avg, neg_avg, samples,
                tissue_name, combined_dir, n_top, sort_by='avg',
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                row_spacing=1.5
            )
        
        if pos_count and neg_avg:
            plot_combined_positive_by_count_negative_by_avg_heatmap(
                gene_corr_dict, pos_count, neg_avg, samples,
                tissue_name, combined_dir, n_top,
                gene_symbol_map=gene_symbol_map if need_mapping else None,
                row_spacing=2.0
            )
    
    elapsed_time = time.time() - start_time
    print(f"\nThe analysis of the {tissue_name} has been completed. The processing time was: {elapsed_time/60:.1f} minutes")
    return True

def main():
    global TARGET_TISSUES
    if not TARGET_TISSUES:
        TARGET_TISSUES = list(set(SAMPLE_TO_TISSUE.values()))
        print(f"Automatically detected tissue type:{TARGET_TISSUES}")
    
    tissue_samples = defaultdict(list)
    for sample in tqdm(SAMPLES, desc="Process the sample"):
        tissue = get_sample_tissue(sample)
        matched_tissue = match_tissue_to_target(tissue, TARGET_TISSUES)
        if matched_tissue:
            tissue_samples[matched_tissue].append(sample)
    
    for tissue in TARGET_TISSUES:
        samples = tissue_samples.get(tissue, [])
        print(f"  {tissue}: {len(samples)} ")
    
    for tissue in TARGET_TISSUES:
        samples = tissue_samples.get(tissue, [])
        if not samples:
            print(f"\n{tissue}:  No sample available")
            continue
        
        analyze_tissue(tissue, samples)

if __name__ == "__main__":
    plt.switch_backend('Agg')
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE + 2
    plt.rcParams['axes.labelsize'] = FONT_SIZE + 1
    plt.rcParams['xtick.labelsize'] = FONT_SIZE - 1
    plt.rcParams['ytick.labelsize'] = FONT_SIZE - 1
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupts the program")
    except Exception as e:
        print(f"\nProgram error occurred: {e}")
        import traceback
        traceback.print_exc()