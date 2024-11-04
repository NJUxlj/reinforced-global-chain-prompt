import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from autocot_utils import fix_seed
