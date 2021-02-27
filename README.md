# SpliceFinder
SpliceFinder, which is a *ab initio* splice site prediction tool, applies convolutional neural network for predicting splice sites on genomic sequences.

## Contains
+ iteration_process/test/iteration/iteration.sh: Used for the iteration process, the inital model is provided.
+ iteration_process/iteration_models: the CNN models generated from the iteration process are provided for application.
+ other_models: The source code for other machine learning models mentioned in the paper.
+ SpliceFinder_sourcecode: CNN used for splice sites prediction.

## Prerequisite
The convolutional neural network is built with Python 3.7.3, Tensorflow 1.13.1 and Keras 2.2.4. Following Python packages should be installed:
+ numpy
+ sklearn
+ Tensorflow
+ Keras

**Samtools** should be installed to obtain the sequence information. 

## Data preparation
To reproduce the iteration process, following files should be prepared:
+ The **fasta file** (Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa) should be downloaded from https://asia.ensembl.org/info/data/ftp/index.html, and put under iteration_process/test/. 
+ The **initial dataset**, which can be downloaded from https://drive.google.com/drive/folders/1P_74FJX06NfLPkq4_qrxhwTRsp2yxkyl?usp=sharing, should be put under iteration_process/test/iteration/train/.
+ The **processed data**, which is used for collecting false positives, can be downloaded from https://drive.google.com/drive/folders/1b8M-OgIyYyr__Rx0XS8uZAeHgFsMuHF_?usp=sharing, and should
be put under iteration_process/generate_seq/.


## Usage
### Iteration process
After the above files are prepared, the iteration process can be started.
```shell
cd SpliceFinder
sh iteration_process/test/iteration/iteration.sh
```
The number of iterations is set to be 100. After the iteration process, the CNN models are saved under iteration_process/test/iteration/.

### Quick run on the toy data
For application, we provide some toy models generated from the iteration process: https://drive.google.com/drive/folders/1erLuzQr9VJaGgtwuTpct2l08ESqQahak?usp=sharing. 
For CNN trained with data of other species 
and multiple species, the models can be downloaded from https://drive.google.com/drive/folders/1xsQtwOWwogj904_LrxdpjZsNyxiD2sKL?usp=sharing.

The toy data can be found here: https://drive.google.com/drive/folders/1kuUSzsYCu1hTeZ3VF12zHB2VL1fZXMrW?usp=sharing. To perform a quick run, first download the toy data 
and put them under SpliceFinder_sourcecode/test, then put the CNN model you'd like to use under SpliceFinder_sourcecode/test and rename the model as CNN.h5 
(we have already put a toy model under this directory), finally run the following command.
```shell
cd SpliceFinder/SpliceFinder_sourcecode/test
python test_Cla.py
```

To generate other test data, run the following command to generate data from human genome randomly and convert the data to desired format:
```shell
cd SpliceFinder/iteration_process/generate_seq
python generate.py
mv dna_loc y_label ../test
cd ../test
sh process.sh
mv encoded_seq y_label ../../SpliceFinder_sourcecode/test/
cd ../../SpliceFinder_sourcecode/test/
python test_Cla.py
```

## Cite us
Welcome to cite our paper.
Wang R, Wang Z, Wang J, et al. SpliceFinder: ab initio prediction of splice sites using convolutional neural network[J]. BMC bioinformatics, 2019, 20(23): 1-13.







