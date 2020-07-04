[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images

A novel method that is able to produce credible handwritten word images by conditioning the generative process with both calligraphic style features and textual content.

![Architecture](https://user-images.githubusercontent.com/9562709/86518741-02a27700-be34-11ea-9f5b-807d6c0ee68f.png)

[GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images](https://arxiv.org/abs/2003.02567)<br>
Lei Kang, Pau Riba, Yaxing Wang, Marçal Rusiñol, Alicia Fornés,  and Mauricio Villegas<br>
Accepted to ECCV2020.

## Software environment:

- Ubuntu 16.04 x64
- Python 3.7
- PyTorch 1.4

## Dataset preparation

The main experiments are run on [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) since it's a multi-writer dataset. Furthermore, when you have obtained a pretrained model on IAM, you could apply it on other datasets as evaluation, such as [GW](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database),  [RIMES](http://www.a2ialab.com/doku.php?id=rimes_database:start), [Esposalles](http://dag.cvc.uab.es/the-esposalles-database/) and
[CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/). 


## How to train it?

First download the IAM word level dataset, then refer your folder in `load_data.py` (search `img_base`). Then run the training with:

```bash
./run_train.sh
```

**Note**: During the training process, two folders will be created: 
`imgs/` contains the intermediate results of one batch (you may like to check the details in function `write_image` from `modules_tro.py`), and `save_weights/` consists of saved weights ending with `.model`.


## How to test it?

We provide two test scripts starting with `tt.`:

* `tt.test_single_writer.4_scenarios.py`: Please refer to Figure 4 of our paper to check the details. At the beginning of this code file, you need to open the comments in turns to run 4 scenarios experiments one by one.

* `tt.word_ladder.py`: Please refer to Figure 7 of our paper to check the details. It's fun:-P


## Citation

If you use the code for your research, please cite our paper:

```
To be updated...
```
