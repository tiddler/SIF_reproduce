# My implement of SIF
This repo contains the code and results for reproducing the results in the paper: *A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS*

As an easy method, it can be used as the baseline for doc similarity matching; sentence embedding(sentence to vector); feature representation; and other tasks. Since it is quite easy and fast, it is recommended to test it before complex neural networks method.

paper link: [A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS](https://openreview.net/pdf?id=SyK00v5xx)

Author's code: [Github](https://github.com/PrincetonML/SIF)

## Requirements

`virtual env` is recommended.

```bash
pip install -r requirements.txt
```

My env:

python==3.6
spacy==2.0.11
numpy==1.14.0
scipy==1.0.0
sklearn==0.19.1

## Algorithm

> SIF
>
> **input:** word embedding $$v_w$$, a set of sentences $$S$$, parameter $$a$$, estimated probabilities $$p(w)$$ of the words
>
> **output:** sentence embeddings $$v_s$$
>
> 1. for all sentence $$s$$ in $$S$$ do
> 2. ​    $$v_s \larr \frac{1}{\lvert s \rvert} \sum_{w \in s} \frac{a}{a+p(w)} v_w$$
> 3. end for
> 4. form a matrix $$X$$ whose columns are $${v_s:s \in S}$$, let $$u$$ be its first singular vector
> 5. for all sentence $$s$$ in $$S$$ do
> 6. ​    $$v_s \larr v_s - u u^T v_s$$
> 7. end for

## Implementation

`data` folder contains the test data for STS14, STS15.

`resources` folder should contain `GloVe` and `PSL` file, please download them and unzip them in the folder. The folder also contains `enwiki_vocab_min200.txt`, which is the word frequency estimation.

> download link: [GloVe](https://nlp.stanford.edu/projects/glove/), [PSL](https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing)

*note: the original PSL file contains some weird words, convert it to UTF-8 to avoiding decoding error.

`SIF` folder contains the code for SIF and the script to generate test results.

## Reproduce Result
```bash
python example.py
```
### Unsupervised Embedding (GloVe)

|         Dataset         | avg-GloVe(paper) | avg-GloVe(reproduce) | GloVe + WR(paper) | GloVe + WR(reproduce) |
| :---------------------: | :--------------: | :------------------: | :---------------: | :-------------------: |
|   deft forum(STS 14)    |       27.1       |       **39.2**       |       41.2        |       **45.9**        |
|    deft news(STS 14)    |       68.0       |       **68.1**       |       69.4        |       **69.5**        |
|    headline(STS 14)     |       59.5       |       **63.9**       |       64.7        |       **67.2**        |
|     images(STS 14)      |       61.0       |       **72.5**       |       82.6        |         82.5          |
|      OnWN(STS 14)       |       58.4       |       **76.3**       |       82.8        |       **85.3**        |
|   tweet news(STS 14)    |       51.2       |       **68.4**       |       70.1        |       **75.2**        |
|        STS14 AVG        |       54.2       |       **64.7**       |       68.5        |       **70.9**        |
|                         |                  |                      |                   |                       |
|  answers-forum(STS 15)  |       30.5       |       **55.6**       |       63.9        |       **72.0**        |
| answers-student(STS 15) |       63.0       |       **65.8**       |       70.4        |         68.0          |
|     belief(STS 15)      |       40.5       |       **57.2**       |       71.8        |       **74.5**        |
|    headline(STS 15)     |       61.8       |       **69.5**       |       70.7        |       **74.8**        |
|     images(STS 15)      |       67.5       |       **75.7**       |       81.5        |       **81.9**        |
|        STS15 AVG        |       52.7       |       **64.8**       |       71.7        |       **74.2**        |



### Semi-Supervised Embedding (PSL)

|         Dataset         | avg-PSL(paper) | avg-PSL(reproduce) | PSL + WR(paper) | PSL + WR(reproduce) |
| :---------------------: | :------------: | :----------------: | :-------------: | :-----------------: |
|   deft forum(STS 14)    |      37.2      |      **46.8**      |      51.4       |        49.1         |
|    deft news(STS 14)    |      67.0      |      **71.7**      |      72.6       |        71.3         |
|    headline(STS 14)     |      65.3      |      **69.1**      |      70.1       |      **71.7**       |
|     images(STS 14)      |      62.0      |      **80.1**      |      84.8       |      **85.2**       |
|      OnWN(STS 14)       |      61.1      |      **80.2**      |      84.5       |      **87.0**       |
|   tweet news(STS 14)    |      64.7      |      **76.9**      |      77.5       |      **79.0**       |
|        STS14 AVG        |      59.5      |      **70.8**      |      73.5       |      **73.8**       |
|                         |                |                    |                 |                     |
|  answers-forum(STS 15)  |      38.8      |      **64.7**      |      70.1       |      **72.4**       |
| answers-student(STS 15) |      69.2      |      **74.1**      |      75.9       |        72.6         |
|     belief(STS 15)      |      53.2      |      **71.3**      |      75.3       |        75.1         |
|    headline(STS 15)     |      69.0      |      **73.9**      |      75.9       |      **77.2**       |
|     images(STS 15)      |      69.9      |      **81.7**      |      84.1       |        83.6         |
|        STS15 AVG        |      60.0      |      **73.2**      |      76.3       |        76.2         |
