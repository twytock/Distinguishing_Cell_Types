# Distinguishing_Cell_Types
Data and Code for "Distinguishing Cell Phenotype Using Epigenotype"

# Overview:
1. Software used:
	1. Python (2.7)
		1. Numpy
		2. Pandas (with the xlrd package installed)
		3. Scikit-learn
		4. Matplotlib
		5. scoop (required for parallelization)
	2. R (3.5.2)
		1. Bioconductor
		2. mclust
    3. PDM
2. Download files from Distinguishing_Cell_Types/
	1. This will unpack the source code, scripts to run the analyses, and the data.
	2. A directory called Distinguishing_Cell_Types/ will be created. It should contain "master_script.sh."
  3. Python scripts contain the source code, and are called by the master_script.sh.
	3. Navigate to this Distinguishing_Cell_Types/. Subdirectories are:
    1. GeneExp/             Directory containing the data and results of the Gene Expression dataset
      1. Data/              Contains metadata, including an .xlsx file of the experimental accesion numbers.
      2. Normalization/     Contains the normalized data as a pandas DataFrame, packaged into a python "pickle" file.
      3. Feature_Selection/ Contains the selected features.
      4. Analysis/          Contains the data needed for generating figures like those in the paper
      5. Figures/           Contains the figures generated from the Analysis/
    2. HiC/                 Directory containing the data and results of the Hi-C dataset
      1. Data/              Contains metadata, including an .xlsx file of the experimental accesion numbers.
      2. Normalization/     Contains the normalized data as a pandas DataFrame, packaged into a python "pickle" file.
      3. Feature_Selection/ Contains the selected features.
      4. Analysis/          Contains the data needed for generating figures like those in the paper
      5. Figures/           Contains the figures generated from the Analysis/
3. Analysis pipeline -- description of master_script.sh
	1. Calculate correlations -- python calculate_correlations.py
    1. Loads data from */Normalization/ and performs the matrix decomposition
  2. Select features        --    python -m scoop -n nproc feature_selection.py [U|W] [U|C] [GeneExp|HiC] N;
    1. "nproc":         *Warning* The default number of processors is 12. Check that your cluster has sufficient resources to run the script. You may choose to tailor the number of processors to your cluster's number of cores/amount of memory.
    2. "[U|W]":         Determines whether weights are applied: U -- "Unweighted"; W -- "Weighted"
    3. "[U|C]":         Determines whether correlations are applied: U -- "Uncorrelated"; C -- "Correlated". The correlated method is only supported if weights are used.
    4. "[GeneExp|HiC]:  Determines whether the GeneExp or HiC dataset is used.
    5. "N"    :         The integer number of features to stop at. It takes ~8 hours to select 10 features on a 48 core cluster on the GeneExp dataset. Beynond N features, the script uses a heuristic to order the remaining features.
    6. Other notes: feature_selection.py is the longest running script. It is *strongly recommended* that you test a small run of the code on your cluster to ensure that it has sufficient memory and processors.
	3. Calculate distances    --    python -m scoop -n nproc calc_dists.py [U|W] [U|C] [GeneExp|HiC] [d|k] N;
		1. "nproc":         *Warning* The default number of processors is 12. Check that your cluster has sufficient resources to run the script. You may choose to tailor the number of processors to your cluster's number of cores/amount of memory.
    2. "[U|W]":         Determines whether weights are applied: U -- "Unweighted"; W -- "Weighted"
    3. "[U|C]":         Determines whether correlations are applied: U -- "Uncorrelated"; C -- "Correlated". The correlated method is only supported if weights are used.
    4. "[GeneExp|HiC]:  Determines whether the GeneExp or HiC dataset is used.
		5. "[d|k]":         Determines the criterion applied: d -- "pairwise *d*istance"; k -- "*k*-nearest-neighbors". Only the "d" criterion is used for the figures.
    6. "N"    :         The integer number of features to include in distance calculations. 
    7. Other notes: calc_dists.py is fast for small N and considerably slower for large N.
  4. Nonconvexity tests     --    python -m scoop -n nproc nonconvexity_tests.py [U|W] [U|C] [GeneExp|HiC];
		1. "nproc":         *Warning* The default number of processors is 12. Check that your cluster has sufficient resources to run the script. You may choose to tailor the number of processors to your cluster's number of cores/amount of memory.
    2. "[U|W]":         Determines whether weights are applied: U -- "Unweighted"; W -- "Weighted"
    3. "[U|C]":         Determines whether correlations are applied: U -- "Uncorrelated"; C -- "Correlated". The correlated method is only supported if weights are used.
    4. "[GeneExp|HiC]:  Determines whether the GeneExp or HiC dataset is used.
    5. Other notes: nonconvexity_tests.py uses 4 features for GeneExp and 3 features for Hi-C by default. This may be adjusted to other values in the "if __name__ == '__main__':" block of the script.
	5. Compare classifiers    --    python -m scoop -n nproc compare_classifiers.py [tiered|loo|tieredpct|loopct];
		1. "nproc":         *Warning* The default number of processors is 12. Check that your cluster has sufficient resources to run the script. You may choose to tailor the number of processors to your cluster's number of cores/amount of memory.
    2. "[tiered|loo|tieredpct|loopct]":   Determines cell type grouping and cross validation method.
      1. "tiered" groups GeneExp experiments by cell type group before applying the LOGO testing of the classifier. It skips the Hi-C dataset.
      2. "loo" applies the LOGO testing of each classifier
      3. "tieredpct" groups GeneExp experiments by cell type group before applying the testing of the classifier on a random selection of experimental groups that are 5% -- 20% of all data.
      4. "loopct" applies the testing of the classifier on a random selection of experimental groups that are 5% -- 20% of all data.
    2. Other notes: compare_classifiers.py will calculate the classification accuracy for the three versions of the KNN method in addition to the SVC/RF methods.
  6. Test correlations      --    python -m scoop -n nproc test_correlations.py [GeneExp|HiC] [1|plot]
		1. "nproc":         *Warning* The default number of processors is 10. Check that your cluster has sufficient resources to run the script. You may choose to tailor the number of processors to your cluster's number of cores/amount of memory.
    2. "[GeneExp|HiC]:  Determines whether the GeneExp or HiC dataset is used.
    3. "[1|plot]"    :         If not N=="plot", integer number for the run number, else the script plots the confusion matrices. The integer must be called first to generate the confusion matrices the script currently assumes that N=1.
  7. Apply HNN method       --    python lang2014_classifier.py
  8. Plot results           --    python plot_overlap.py
    1. Other notes: The plotting routine is still being tested. Double check to ensure that the necessary results have been generated and that the files are correctly referenced in the script.
  9. Other analysis         --    Analysis.ipynb
    1. To run the analysis notebook, jupyter/ipykernel need to first be installed.
    2. Then type "jupyter notebook" in your command line, a menu will open in your web browser.
    3. Find "Analysis.ipynb" in the window and double click on it to open it.
    4. Executing each cell in order should generate the figures listed in the headers.
    5. Other notes: The plotting routine is still being tested. Double check to ensure that the necessary results have been generated and that the files are correctly referenced in the script.
 
# Further comments

## Software Used
We recommend using a package manager like [conda](https://www.anaconda.com) or [pip](https://pypi.org/project/pip/) to install python packages. All packages should be installable using a recipe like ```conda install <pkg-name>``` or ```pip install <pkg-name>```. If you do not have administrative access on your machine, you may want to use ```pip install <pkg-name> --user```, which installs packages in the user’s local directory (~/.local/) instead of at the root level (/usr/local/). Before using the newly installed packages, we recommend logging in and out so that any relevant $PATH variables are updated.

Above, we use angle brackets <> to denote quantities in quotations that may take on values defined by the user e.g., <num-procs> can be set to an appropriate value for a machine. By default <num-procs>=8. 

