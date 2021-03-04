"""
Script for combining multiple pose labelled files (creating multi-scene files)
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("scenes_path",
                            help="path to a folder with scene labels files")
    arg_parser.add_argument("out_file",
                            help="name of output file, combining all scenes in the input path")
    args = arg_parser.parse_args()
    labels_files = [join(args.scenes_path, f) for f in listdir(args.scenes_path) if isfile(join(args.scenes_path, f))]
    dfs = [pd.read_csv(labels_files[i], converters={'scene':str}) for i in range(len(labels_files))]
    combined_df = pd.concat(dfs)
    combined_df.to_csv(args.out_file, index=False)