#!/usr/bin/python3

import argparse
from scipy.io import savemat, loadmat
import numpy as np
from preprocessing import preprocessing
from algorithm import FIST
from postprocessing import postprocessing
from report import report
import os
import json
import sys


def main():

    # parse the arguments from the command line
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_path")
    parser.add_argument("--preprocessing", action="store_true")
    parser.add_argument("--algorithm", action="store_true")
    parser.add_argument("--postprocessing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--visium", action="store_true")
    parser.add_argument("--nodiscrete", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("-p", "--ppi", 
        help="Path to a protein-protein interaction network")
    parser.add_argument("-n", "--binning", 
        help="Dimensions for the binning of the spots")
    parser.add_argument("-g", "--geneformat", 
        choices=["symbol", "entrez", "ensembl"], default="symbol",
        help="Format of the gene names/symbols")
    parser.add_argument("-o", "--organism", default="human", 
        help="Organism the data belongs to")
    parser.add_argument("-gp", "--geneplot", default="",
        help="Genes of interest to plot in final report, seperated by commas.")
    parser.add_argument("-l", "--lam", type=float, default=0.1, 
        help="The hyperparameter lambda")
    parser.add_argument("-k", "--rank", type=int, default=200, 
        help="Rank of the tensor")
    parser.add_argument("-s", "--stopcrit", type=float, default=1e-4, 
        help="Stopping criteria")
    parser.add_argument("-i", "--maxiters", type=int, default=500,
        help="Maximum num. of iterations")
    parser.add_argument("-v", "--validation", type=float, default=None,
        help="Percent of data to hold out for validation")
    parser.add_argument("-r", "--seed", type=int,
        help="Random seed for validation")
    parser.add_argument("-b", "--backend",
        choices=["numpy", "cuda"], default="numpy", 
        help="Backend for Tensorly")
    parser.add_argument("-m", "--metadata",
        help="Metadata for postprocessing")
    parser.add_argument("-of", "--outformat",
        choices=["csv", "parquet"], default="csv",
        help="Output file type")
    # fmt: on

    args = parser.parse_args()
    num_modes = sum([args.preprocessing, args.algorithm, args.postprocessing])

    if num_modes > 1:
        raise ValueError(
            "FIST can only be run in 1 mode at a time. To run all modes at once, run without flags."
        )

    # binning should not happen with visium data because it is pre-binned
    if args.binning and args.visium:
        raise ValueError("10X Visium data should not be binned.")

    # this should set the seed across all runs
    if args.seed is not None:
        np.random.seed(args.seed)

    # if they didn't specify metadata, check to see if it's in the same path as the other input
    if args.metadata is None:
        try_metadata = os.path.join(os.path.split(args.input_file)[0], "metadata.json")
        if os.path.exists(try_metadata):
            args.metadata = try_metadata

    # create the output path if need be
    if not (os.path.exists(args.output_path)):
        print(f"Output path {args.output_path} does not exist, creating...")
        os.makedirs(args.output_path)

    if args.preprocessing and num_modes == 1:
        T, W_g, metadata, val_indices, val_values = preprocessing(
            args.input_file,
            args.ppi,
            args.visium,
            args.binning,
            args.nodiscrete,
            args.validation,
            args.geneformat,
            args.organism,
            args.geneplot,
            args.rotate,
            args.verbose,
        )
        if args.verbose:
            print(f"Writing tensor to output file...", flush=True)
        savemat(
            os.path.join(args.output_path, "tensorPPI.mat"),
            {"T": T, "W_g": W_g, "VI": val_indices, "VV": val_values},
        )
        metadata["arguments"] = " ".join(sys.argv)
        f = open(os.path.join(args.output_path, "metadata.json"), "w")
        f.write(json.dumps(metadata))
        f.close()
    if args.algorithm and num_modes == 1:
        data = loadmat(args.input_file)
        T = data["T"]
        W_g = data["W_g"]
        val_indices = data["VI"]
        val_values = data["VV"]
        f = open(args.metadata, "r")
        metadata = json.load(f)
        f.close()
        T_imputed = FIST(
            T,
            W_g,
            l=args.lam,
            rank_k=args.rank,
            stop_crit=args.stopcrit,
            max_iters=args.maxiters,
            val_indices=val_indices,
            val_values=val_values,
            verbose=args.verbose,
            metadata=metadata,
            backend=args.backend,
        )
        savemat(
            os.path.join(args.output_path, "imputedtensor.mat"),
            {"T_imputed": T_imputed},
        )
        f = open(os.path.join(args.output_path, "metadata.json"), "w")
        f.write(json.dumps(metadata))
        f.close()

    if args.postprocessing and num_modes == 1:
        data = loadmat(args.input_file)
        T_imputed = data["T_imputed"]
        f = open(args.metadata, "r")
        metadata = json.load(f)
        f.close()
        df = postprocessing(T_imputed, metadata, args.verbose)
        if args.verbose:
            print(f"Writing data to .{args.outformat} format...", flush=True)
        if args.outformat == "parquet":
            df.to_parquet(
                os.path.join(args.output_path + "/FIST_output.parquet"), index=True
            )
        else:
            df.to_csv(os.path.join(args.output_path + "/FIST_output.csv"))
        if args.report:
            if args.verbose:
                print("Creating report...")
            report(os.path.join(args.output_path + "/FIST_report.pdf"), metadata)

    if num_modes == 0:
        T, W_g, metadata, val_indices, val_values = preprocessing(
            args.input_file,
            args.ppi,
            args.visium,
            args.binning,
            args.nodiscrete,
            args.validation,
            args.geneformat,
            args.organism,
            args.geneplot,
            args.rotate,
            args.verbose,
        )
        metadata["arguments"] = metadata["arguments"] = " ".join(sys.argv)
        T_imputed = FIST(
            T,
            W_g,
            l=args.lam,
            rank_k=args.rank,
            stop_crit=args.stopcrit,
            max_iters=args.maxiters,
            val_indices=val_indices,
            val_values=val_values,
            verbose=args.verbose,
            metadata=metadata,
            backend=args.backend,
        )
        df = postprocessing(T_imputed, metadata, args.verbose)
        if args.verbose:
            print(f"Writing data to .{args.outformat} format...", flush=True)
        if args.outformat == "parquet":
            df.to_parquet(
                os.path.join(args.output_path + "/FIST_output.parquet"), index=True
            )
        else:
            df.to_csv(os.path.join(args.output_path + "/FIST_output.csv"))
        if args.report:
            if args.verbose:
                print("Creating report...")
            report(os.path.join(args.output_path + "/FIST_report.pdf"), metadata)
    if args.verbose:
        print("Done!")


if __name__ == "__main__":
    main()
