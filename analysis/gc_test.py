import sys
import os 
sys.path.append(os.path.abspath('../'))

from analysis.gc_tools import create_service_account_credentials
from analysis.load_dfs import load_post_count_ili

print(load_post_count_ili('fr', create_service_account_credentials()))