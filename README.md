# Descent-to-Delete

Code accompanying the following paper:

> [Descent-to-Delete: Gradient-Based Methods for Machine Unlearning](https://arxiv.org/abs/2007.02923)\
> Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi\
> _arXiv:2007.02923_

Read my blog post on the topic [here](https://chriswaites.com/#/machine-unlearning) for a detailed explanation!

## Getting Started

If you're familiar with conda:

```
conda env create -f environment.yml
```

Then activate your environment via:

```
conda activate data-deletion
```

Alternatively, you can install Python 3.8 and the necessary dependencies via:

```
pip install -r requirements.txt
```

## Example

```
python logistic_regression.py
```
