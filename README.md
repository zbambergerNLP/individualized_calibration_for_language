<h1 align="center"> Individualized Calibration For Encoder Language Models </h1> 

  <p align="center">
    <a href="https://github.com/zbambergerNLP"> Zachary Bamberger </a> •
    <a href="https://github.com/idankinderman"> Edan Kinderman </a>
  </p>


<h2 id="summary"> :book: Summary </h2>

Bias detection and mitigation in Natural Language Processing (NLP) pose significant challenges, particularly in addressing individual instances and nuanced forms of bias that transcend predefined protected attributes. Building upon the work of Zhao et al., (2020) [[1]](#ref1), we introduce Individualized Random Calibration for Language Forecasters (IRCLF), a new method that integrates individually calibrated random forecasters with empirical transformer encoder models [[2]](#ref2). Unlike conventional methods, IRCLF seamlessly balances accuracy and \textit{individualized} fairness during the training phase, without dependence on predefined demographic information.

We leverage IRCLF for the toxicity prediction task [[3]](#ref3) [[4]](#ref4), measuring the model's learned bias towards nine protected groups. Our findings reveal a successful replication of the trends in [[1]](#ref1), where a single hyper-parameter, $\alpha$, adeptly negotiates between an accuracy-oriented loss (NLL) and a fairness-oriented loss (PAIC). Yet, a critical examination employing more expressive and practical classification metrics uncovered shortcomings in achieving fairness by contemporary standards. This work not only advances the understanding of individualized fairness in NLP but also identifies vital areas for future research and improvement


<br />

## Requirements
- pytorch 2.0 or higher with cuda 11.8
- transformers 4.0 or higher
- numpy
- pandas
- sklearn
- tqdm
- matplotlib
- accelerate
- datasets

## Usage
### 1. Download the data from Kaggle [[4]](#ref4)
### 2. Set up the accelerated training configuration
`accelerate config`
### 3. Run the training script

The primary training script to run is `main2.py`. 

The script can be run with the following command:

`accelerate launch main2.py`

Note the flags that can be set in the script, which are described in `flags.py`.


<br />


<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* <a id="ref1">[[1]](https://arxiv.org/abs/2006.10288)</a> "Individual calibration with randomized forecasting", Zhao, Shengjia and Ma, Tengyu and Ermon, Stefano. International Conference on Machine Learning.
* <a id="ref2">[[2]](https://arxiv.org/abs/1810.04805)</a> "Bert: Pre-training of deep bidirectional transformers for language understanding", Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina, 2018.
* <a id="ref3">[[3]](https://arxiv.org/abs/1903.04561)</a> "Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification", Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy. Association for Computing Machinery, 2019.
* <a id="ref4">[[4]](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv)</a> Jigsaw Unintended Bias in Toxicity Classification, Kaggle.
