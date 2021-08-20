# GSoC (2021) - Gene Pair Orthology Prediction Using Deep Learning

This project forms part of Google Summer of Code (GSoC) 2021 and is in collaboration with the Compara Project at the European Bioinformatics institute (EBI). All work in this repository was completed by myself as part of the GSoC, with feedback from colleagues at the EBI, and serves as a stable record of the work that was completed during GSoC 2021. Further work will continue after GSoC 2021, and will be incorporated in the main [Compara GSoC](https://github.com/EnsemblGSOC/compara-deep-learning) github repository on a continued basis.

The primary objective of the project was achieved within the specified time of GSoC, namely to establish that deep learning is capable of inferring the homology relationship of gene pairs with high accuracy. This was done using a set of convolutional neural networks, as detailed below, which were implemented using TensorFlow2.

Further work to continue beyond this project are to: (i) establish a "push-button" solution, capable of categorising a query gene pair using a trained network with a single command line argument; (ii) elaborate upon when precisely the networks miss-classifies gene pairs; and, (iii) to identify which, if any, of the input features to the network can be removed without being detrimental to the network's final performance.

In the rest of this readme I will elucidate the project's goals, describe the current network architecture, and explain the network's input along with the pre-processing pipeline used to generate this input.

## Project Goal

The aim of this project is to take pairs of genes from different species and predict the nature of their evolutionary relationship. Primarily, this concerns whether the genes are orthologs or paralogs (see: [here](https://useast.ensembl.org/info/genome/compara/homology_types.html)). The compara project at the EBI currently uses a method that takes advantage of sequence similarity and the species tree to categorise gene pairs. Whilst it has proven successful in the past, the current method does not scale well as the number of comparison pairs increases. Thus, as the genomes of more species are sequenced, the ability to assess the homology between genes is inhibited by the computational resources required for this task. 

This project is a proof of principle that <em>deep learning</em> is capable of predicting homology relationships with a high degree of accuracy between even distant species, with a view to demonstrate it's viability for predicting homology at scale in a production setting. 

## The network's input

In the project, a neural network capable of attaining > 99% validation set accuracy was attained using a <em>convolutional neural network</em> which used a variety of features in the form of matrices as well as the distance between species to form the basis of it's predictions.

Every input matrix represents the genes in the neighborhood of the two genes being compared. Let's call the comparison genes, whose homology we wish to infer, `A` and `B`. In the current implementation, the 3 genes upstream and the 3 genes downstream of both genes `A` and `B` on their respective chromosomes are obtained, giving two vectors of 7 genes. For instance, the vector representing gene `A` and its neighbours can be written as `v1 = [d3,d2,d1,A,u1,u2,u3]`. Here, the `d` elements represent the downstream genes, `A` represents gene `A` itself, and the `u` elements represent the upstream genes. `v2` is likewise formed for gene `B`. All pairwise combinations of genes between these two vectors are then made, and then stored in a 7x7 matrix, called `M`. As such, element `i,j` of `M` represent some comparison between gene `i` from `v1` and gene `j` from `v2`. Several different comparison mertics are used, resulting in multiple matrices, each housing different metrics of comparison between the genes in the neighbourhood of genes `A` and `B`. 5 such types matrices are formed which are meant to capture evolutionary information relevant to the development of classifying the homology relationship. These feature matrices are briefly as follows:

1) The global alignment matrix
2) The local alignment score matrix
3) The local alignment coverage matrix
4) The Pfam Jaccard matrix
5) The Pfam coverage matrix

Further details of each of these matrices can be found in the steps detailing the [pipeline](#the-pipeline).

## The network

For each type of these feature matrices, the network has a separate CNN sub-module that learns a feature map for that matrix (each feature gets its own CNN because the features are so diverse there is no reason to for them to apply the same filters to each of these layers. All features are later joined in MLP layers to merge all of them together). Each CNN sub-module is visualised immediately below. The output of the sub-module is a fully connected MLP layer derived from the earlier CNN matrix values. 

![alt text](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/CNN_model.png)

The output for each sub-module is then concatenated to and fed into an MLP layer as shown below, which in turn is concatenated with the value of the species distance between the species containing the two comparison genes. All of these are then either fed into a two or three way softmax layer which is used to assign probabilities for the gene pairs homology category. The two way classification could simply be ortholog vs paralog, whilst a three way classification task would be ortholog vs paralog vs non-homologous. The nodes in the first layer of the network are coloured by their feature type, and by extension, the CNN sub-module that they originated from (with only 5 input nodes representing the 50 for cleaner visualisation).

![alt text](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/MLP_layers.png)


## Snakemake implementation

The workflow manager selected has been [snakemake](https://snakemake.readthedocs.io/en/stable/index.html) and makes use of a conda environment stored in a [YAML](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/compara.yml) file in this repo to ensure as much reproducibility as possible. Moreover, once this conda environment is installed, all packages required for running the pipeline from end to end should be installed as well.

Snakemake allows to specify a sequence of <em>rules</em> that can be used to transform input files into output files. By stringing many such rules together, snakemake allows to elegantly formulate an entire data processing pipeline. Whilst the sequence of rules and transformations can be challenging to understand by reading a pipeline's script, the Directed Acyclic Graph (DAG) of jobs that are used to go from raw input to output can be visualised to aid in understanding the pipelines steps. An example of this can be seen here:

![alt text](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/small_dag.png)

At first, this can be a little intimidating to view, however after a little thought it can be seen how much this visualisation aids in understanding the process by informing us which rules in the pipeline are dependent on which others before they run. An arrow from one rule to the next indicates the sequence in which a rule must be run. Moreover, once a rule has been established, it can further be generalised across multiple examples. For instance, above, the DAG is demonstrated for <em>neolamprologus brichardi</em>. But we can easily run the pipeline for multiple species, for example, the dag below generalises this to 3 species. We can in principle generalise the pipeline to as many species as we would like, provided that we have the necessary raw data and compute power to execute the pipeline. For the current demonstration, the pipleine was run with 56 species(the DAG can be seen [here](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/dag.png), though the image is very large).

![alt text](https://github.com/AidanMar/GSoC-2021-compara-deep-learning/blob/master/medium_dag.png)

## The pipeline

### Pre-requisites
If you clone the repository into your cluster, you should be able to run it after some initial set-up. Note that the code presented has been tested in EBI's LSF cluster Codon, so any differences in configuration may affect the software's functionality or performance.

To get going, make sure you have access to _anaconda3_ or _miniconda_ so that you can work with conda environments. Firstly, clone this git repository to a suitable place in your cluster system. Once this has been completed, install the conda environment specified by the `compara.yml`. To create the environment, simply run:
```
 conda env create -f compara.yml
```
There are a lot of packages required to run this pipeline from end to end, so do be patient with the installation process. For more info on dealing with conda environments and YAML files see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Setting up Diamond

If you have installed the conda environment, you have installed [Diamond](https://github.com/bbuchfink/diamond) as well. This is the tool used to align genes to the PFAM-A database and is used in the pipeline to generate a database of alignments for each species. However, before this can be done, Diamond requires some pre-setup to get going. Go to the EBI's [FTP](http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/) and download `Pfam-A.fasta.gz`. Then run:
```
diamond makedb --in /path1/Pfam-A.fasta.gz -d /path2/Pfam-A
```
where the inputs and outputs are modifiable to wherever is most appropriate on your system.

### Configuration file

Now that the package environment is set-up, we need to specify the configuration file for the environment. This is the `config.json` file. By modifying this file, most of the flow of information through the pipeline should be readily controlled.

- `species_file`: contains the list of species that we wish to sample homologous pairs from
- `reports_out_base_path`: the path to the directory that will contain all the subdirectories to which a `stdout` will write for each of the jobs that snakemake submits to the cluster. For instance, the rule `split_diamond_alignment` will submit jobs to the cluster and then write its `stdout` to `config["reports_out_base_path"] + "/Diamond_Split/"`
- `num_neighbours`: controls the number of upstream and downstream genes used to generate the feature matrices. This will produce `(2n + 1) x (2n + 1)` feature matrices.
- `species_neighbour_outdir`: path to directory of where to store neighbour genes as a dictionary for each species
- `Longest_pep_fasta_path`: path to directory of where to store the longest peptide sequences of every gene
- `Diamond_alignments_path`: path to directory of where to store the output of the Diamond alignments
- `Diamond_Pfam_db_path`: path to the Pfam-A database generated in the Diamond configuration step above, i.e. wherever you put `/path2/Pfam-A`
- `samples_neighbour_outdir`: path to the directory containing all the gene which are randomly sampled from the homology databases
- `samples_per_species`: the number of samples you wish to get from each species homology database
- `num_homolog_species`: the number of homology species to attempt to get from each database
- `sp_names`: path to the file containing names of the species in the species distance matrix (included in this repository)
- `dist_matrix`: path to the file containing the species distance matrix (included in this repository)
- `out_dir`: path to directory where you would like most of the output information to go to
- `Final_data`: path to the directory where the final compiled and concatednated will be save. These files are ~1-10Gbs in size.
- `Model_outputs`: path to the directory where the saved models will be saved as well as their evaluation statistics

### Configuration directory

Within this repo there is a configuration directory called `config`. This has a set of files, each one containing the file paths to files needed to run this pipeline. This was run natively on the EBI server, and therefore all files are readily available on the system. If you're running this on another system, then you'll need to download these files before running this pipeline.

The three main file types are the <em>peptide</em> files which contain the protein sequences, the <em>CDS</em> files which hold the coordinates of the protein coding sequences of each genes, and the <em>homology database</em> files which contain the list of labelled homologous pairs for each species that the EBI has data from. In order to run this pipeline for a species, you need to have all three such files for your species of interest. If not, it cannot be run. By investigating the species which have all three files present, you can compile a list of species you wish to run the pipeline for. I did this, and placed a completed list into `config/valid_species.txt`. My selection procedure was relatively random and I just picked ~50 species that had all the relevant data. You can generate your own species set if you're interested in other species.

### Running the pipeline

Everything in the pipeline is controlled by the snakefile. Each rule of the snakefile can be seen earlier in the repository. Here is described what each rule does, their order in the DAG, and what script in the `scripts/` folder is required for each rule. Whilst snakemake implements each of these rules in turns, it simply strings together a sequence of bash and python scripts which are modularised such that one can in principle take the script and run each step separately without snakemake, if desired. Some modifications to the input and output formats for each script would be required to enable this to happen.

#### rule: select_longest

Inspects each species FASTA file, selects the longest version of each gene and writes it to a knew FASTA file.

script: [GeneReferencePetpideSelection.py](scripts/GeneReferencePetpideSelection.py)

#### rule: Diamond_alignment

Takes each of the fasta files output from the rule<em>select longest</em> and aligns each genes for that species to the Pfam-A database to identify all of the Pfam domains within each gene. The output is a TSV file containing each gene, and the Pfam domain that it aligns to.

script: uses Diamond which is part of the `compara.yml` conda environment

#### rule: split_diamond_alignment

Each of the TSV files output from a diamond alignment can be very large, too large to hold in memory. Therefore, each TSV file is split into many separate TSV files, each containing the Pfam alignments for each gene, as these can easily be loaded into memory one at a time.

#### rule: species_neighbour

Goes through every gene in a species genome using the species' GTF file, and finds its upstream and downstream neighbouring genes. These are then written to a dictionary. This is done serparately for each species.

script: [geneNeighbourFormater.py](scripts/geneNeighbourFormater.py)

#### rule: select_samples

Reads the homology TSV file of the species, and selects homologous gene pairs between the species the homology file belongs to and multiple other "query species". There are many possible ways to select candidate gene pairs from this TSV file. The current implementation reads in the `valid_species.txt` file, and only allows comparison between species in that list. The potential query species are then ranked in order of their species distance, as stated in the species distance matrix. `n` query species are then chosen at evenly spaced ranks, where `n` is controlled in the `config.json` by `num_homolog_species`. The rule aims to sample roughly equal numbers of orthologs and paralogs for comparison. Non-homologous genes are also randomly sampled between this set of species to give a list of gene pairs and their orthology type labels of `["ortholog","paralog","not"]`. This is not the only way to generate a candidate gene pairs dataset, but the random sampling should prevent forms of bias slipping into the dataset. The script mostly exploits pandas to do the heavy lifting. The output pair names, along with some other features about the pairs, are saved as CSV files at the location specified by `samples_neighbour_outdir`. Additionally, each sampled gene's neighbourhood is identified by querying the gene names against the dictionaries produced by the rule <em>species_neighbour</em>. These are saved in `{SPECIES}_sample_neighbour.json` as dictionaries. Similarly, this is done for the each gene's `homolog {SPECIES}_homology_sample_neighbour.json`. The "negative" examples are those that are not homologous pairs, but just randomly sampled, unrelated pairs of genes. These are saved at `{SPECIES}_negative_sample_neighbour.json` and `{SPECIES}_negative_homology_sample_neighbour.json` respectively. 

script: [sample_gene.py](scripts/sample_gene.py)

#### rule: synteny_matrix & negative_synteny_matrix

This rule produces the first set of feature matrices that will in turn be fed into the network. The index of each element in the matrix was described [earlier in this README](#the-network). We use the same notation here. These matrices are saved with the following names as numpy's NPY files:
```
{SPECIES}_global1.npy
{SPECIES}_global2.npy
{SPECIES}_local1.npy
{SPECIES}_local2.npy
```
Gene comparisons of `global1` are made by taking the normalised levenstein matrix distance between each gene `i` and gene `j`. Matrix `global2` is identical, except the sequences of gene `j` is now reversed. 

`local1` in fact contains 3 matrices, stacked one on top of another like the RGB channels of an image. The first of these contains a normalised Smith-Waterman alignment score between the compared genes. The next matrix contains the normalised coverage of compared gene `i` against compared gene `j`. The final matrix contains the normalised coverage of gene `j` against gene `i` as the coverage operation is not symmetric. `local2` is identical to `local1` except that gene `j` has its sequence reversed.

The "negative" versions of these files are identical to those described above, except it is ran for those pairs of genes that were identified as being non-homologous/unrelated in the <em>select_samples</em> step.

script: [synteny_matrix.py](scripts/synteny_matrix.py) & [negative_synteny_matrix.py](scripts/negative_synteny_matrix.py)

#### rule: pfam_matrix & negative_pfam_matrix

This set of matrices utilises the Diamond alignments and the Pfam-A, and therefore requires both sample genes and the sequence of steps relating to diamond to have been run. This is reflected in the earlier DAG. 

The output is saved as `{SPECIES}_negative_Pfam.npy`. This contains 3 matrices stacked like an RGB channel. These Pfam matrices are aimed to contain information regarding the kind of functional structural domains that are shared between two genes, which should enable orthologs and paralogs to be differentiated from one another. Earlier in the pipeline, we found the list of Pfam domains that each gene aligned, and the positions of those alignments, using Diamond. When comparing gene `i` and gene `j` at this step, it takes the set of Pfam-domains each gene aligns to, along with the coordinates of those alignments. It then computes the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) between these sets where, in order for a Pfam gene to be in both sets, both genes must have aligned to it (and at an overlapping position). A Jaccard index of 1 indicates that the genes share lot of functional domains, whilst a Jaccard index of 0 means that no such shared domains are present, and this should be reflective of the different natures of the evolitionary relationships between different gene pairs. This information forms the first matrix in `pfam.npy`.

Whilst likely useful, the Jaccard index as a scalar value is being asked to represent a lot of degrees of variation in the range of functional domains shared between evolutionary relationships. The next two matrices in `pfam.npy` aim to rectify this by taking the coverage overlap between the two pairs of sequences, which aims to capture variable information that might reflect, for instance, disparities in the lengths of the two genes. Again, the coverage of A on B is not the same as B on A, so two matrices are requried to represent this in both directions.

Once more, this is done separately for the "negative"/non-homologous pairs, by using a different script.

script: [Pfam_matrix_parallell_split.py](scripts/Pfam_matrix_parallell_split.py) & [negative_Pfam_matrix_parallel_split.py](scripts/negative_Pfam_matrix_parallel_split.py)

#### rule: format_data

Takes all the output matrices for each species, the `samples.csv` files and combines them into a single large dataset. The final datasets are labelled by date to keep track of progress. 

#### rule: train_model

Trains a neural network architecture to predict the homology classes. The model is written in TensorFlow2 using a combination of the Sequential and Functional APIs. The Sequential API is straightforward to use, but when non-standard details of the network need to be specified, the Functioanl API is required. 

The trainer uses dataset generators to train the model, where only subsets of the data are read from disc at anyone time. This is because holding the entire, large training datasets in memory can be prohibitively costly. Whilst training, we iterate through the dataset, directly from disc. The model is then saved to the "Model_outputs" parameter specified within the `config.json` file.

script: [trainer.py](scripts/trainer.py)

#### rule: evaluate_model

Using this we evaluate the model's performance and save the figures at the specified output directory.
