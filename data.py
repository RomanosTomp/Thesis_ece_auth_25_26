import numpy as np
import argparse
import os
import pandas as pd


def load_data(datafile):
    """Load data from a CSV file and return features and labels."""
    data = pd.read_csv(datafile,
                       index_col='time'
                       , parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotations': 'string'}
    )
    return data

