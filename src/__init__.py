import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')

CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(CACHE_DIR, 'hf_cache'))
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR

ADAPTATION_DATA_DIR = os.path.join(DATA_DIR, 'adaptation')
RAW_ADAPTATION_DATA_DIR = os.path.join(ADAPTATION_DATA_DIR, 'raw')
PROCESSED_ADAPTATION_DATA_DIR = os.path.join(ADAPTATION_DATA_DIR, 'processed')

TUNING_DATA_DIR = os.path.join(DATA_DIR, 'tuning')
RAW_TUNING_DATA_DIR = os.path.join(TUNING_DATA_DIR, 'raw')
PROCESSED_TUNING_DATA_DIR = os.path.join(TUNING_DATA_DIR, 'processed')

METADATA_DATA_DIR = os.path.join(DATA_DIR, 'metadata')
RAW_METADATA_DATA_DIR = os.path.join(METADATA_DATA_DIR, 'raw')
PROCESSED_METADATA_DATA_DIR = os.path.join(METADATA_DATA_DIR, 'processed')