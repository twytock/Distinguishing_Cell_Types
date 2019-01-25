## reconstitute data
python split_and_reconstitute_large_files.py

## calculate correlations
python calculate_correlations.py;

## select features
python -m scoop -n 12 feature_selection.py W C GeneExp 10;
python -m scoop -n 12 feature_selection.py W C HiC 10;
python -m scoop -n 12 feature_selection.py W U GeneExp 10;
python -m scoop -n 12 feature_selection.py W U HiC 10;
python -m scoop -n 12 feature_selection.py U U GeneExp 10;
python -m scoop -n 12 feature_selection.py U U HiC 10;

## calculate distances
python -m scoop -n 12 calc_dists.py W C GeneExp d 4;
python -m scoop -n 12 calc_dists.py W C HiC d 3;
python -m scoop -n 12 calc_dists.py W U GeneExp d 4;
python -m scoop -n 12 calc_dists.py W U HiC d 3;
python -m scoop -n 12 calc_dists.py U U GeneExp d 4;
python -m scoop -n 12 calc_dists.py U U HiC d 3;

## nonconvexity tests
python -m scoop -n 12 nonconvexity_tests.py W C GeneExp;
python -m scoop -n 12 nonconvexity_tests.py W U GeneExp;
python -m scoop -n 12 nonconvexity_tests.py U U GeneExp;
python -m scoop -n 12 nonconvexity_tests.py W C HiC;
python -m scoop -n 12 nonconvexity_tests.py W U HiC;
python -m scoop -n 12 nonconvexity_tests.py U U HiC;

## compare classifiers (data for Table I)
python -m scoop -n 12 compare_classifiers.py tiered;
python -m scoop -n 12 compare_classifiers.py loo;
python -m scoop -n 12 compare_classifiers.py tieredpct;
python -m scoop -n 12 compare_classifiers.py loopct;

## test_correlations
python -m scoop -n 10 test_correlations.py GeneExp 1
python -m scoop -n 10 test_correlations.py HiC 1
python test_correlations.py GeneExp plot
python test_correlations.py HiC plot

## Lang 2014 method
python lang2014_classifier.py

## plot figures
python plot_overlap.py

