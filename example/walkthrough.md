# FIST-GT Walkthough

Simlulated data is provided in this folder as `example_data.csv` and `example_PPI.txt`. If FIST was downloaded from source, navigate to the `example/` directory. If not, download the files from [this](https://github.com/kuanglab/FIST-nD/tree/master/example) directory to continue.

We first remove the precompiled output files.

```
rm example/FIST_output.csv && example/rm FIST_report.pdf
```

Now, we run the entire FIST analysis with one command.

If installed from pip:

```
python3 -m fistnd example/example_data.csv example/ --verbose --report -p example/PPI.txt -n 20x20x20 -gp A,B,C,D,E,F,G,H,I,J -r 0 -l 0.01 -v 0.1
```

If installed from source:

```
python3 FIST.py example/example_data.csv example/ --verbose --report -p example/PPI.txt -n 20x20x20 -gp A,B,C,D,E,F,G,H,I,J -r 0 -l 0.01 -v 0.1
```

Breaking down the arguments:
* `example/example_data.csv`: The input data file.
* `example/`: The output directory.
* `--verbose`: Verbose output.
* `--report`: Create a report summarizing the imputation.
* `-p example/PPI.txt`: The PPI used for connections.
* `-n 20x20x20`: Use a tensor size of at most 20x20x20.
* `-gp A,B,C,D,E,F,G,H,I,J`: Include a plot of the genes A, B, ..., J in the final report.
* `-r 0`: Set the random seed to 0.
* `-l 0.01`: Use 0.01 as a valdation parameter for λ.
* `-v 0.1`: Hold out 10\% of the data to use as a crossvalidation set.

After running this command, there should be two more files in the `example/` directory, `FIST_output.csv` and `FIST_output.pdf`. We should get output like the following:

```
Reading input data...
Binning and filtering coordinates...
dim 3 appears to be discrete. Converting to discrete.
Reading PPI...
Genes kept: 10 of 10 (100.0%)
PPI connections used in model: 24 (2.4 connections per gene)
Creating tensor of shape (10, 20, 20, 4)...
Performing validation split...
FIST Iter 0     Res: 0.43923    MAE: 0.44791    SMAPE: 0.21562  R^2: 0.10234
FIST Iter 1     Res: 0.09448    MAE: 0.37042    SMAPE: 0.18753  R^2: 0.38891

...

FIST Iter 19    Res: 0.01249    MAE: 0.31682    SMAPE: 0.16481  R^2: 0.55726
Rescaling coordinates...
Finding interpolation coefficients...
Computing interpolated expression at each spot...
Writing data to .csv format...
Creating report...
Done!
```
