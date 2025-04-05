This is my project package for my genomic optimization experiments for STA410. 

To get started with the tutorial.ipynb notebook, you'll need a gpu machine, as well as a python environment with python=3.9

Install the project package with `pip install -e . ` This should install all the dependencies as well.

Then you will need to download model weights of the pretrained model we're using along with 
a hg38 fasta file (the human genome sequence).

wget "https://zenodo.org/records/14604495/files/GATA2.torch?download=1" -O GATA2.torch
wget "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" -O hg38.fa.gz
gunzip hg38.fa.gz

Then get started with the tutorial notebook.

Note!!
If you run into problems with pyfaidx (if you see an error message to do with hg38.fa, use the provided hg38 index file, hg38.fa.fai)
