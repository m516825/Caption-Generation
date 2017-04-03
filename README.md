Caption-Generation
====
Caption generation using seq2seq model with LSTM cell

## Environment
python3 <br />
tensorflow 1.0 <br />

## Usage 
Download hw2 data from kaggle, and GloVe 300 dim <br />
<br />
./Caption-Generation/MLDS_hw2_data/* <br />
./Caption-Generation/MLDS_hw2_data/glove/glove.6B.300d.txt

## Train
First time use, you need to do the preprocessing
```
$ python3 caption_gen.py --prepro 1
```
If you already have done the preprocessing
```
$ python3 caption_gen.py --prepro 0
```
## Model
There are three different model available, including **CaptionGeneratorBasic** <br />
**CaptionGeneratorMyBasic**, and **CaptionGeneratorSS** <br />

1. CaptionGeneratorBasic
  * greedy inference
2. CaptionGeneratorMyBasic
  * beam search
  * greedy inference
3. CaptionGeneratorSS
  * schedule sampling
  * beam search
  * greedy search

You can set model_type to new different model. e.g.
```
$ python3 caption_gen.py --prepro [1/0] --model_type=CaptionGeneratorSS
```

## Inference 
This code provide two inference methods, **Greedy Search** and **Beam Search** <br />
beam search inference is not available in CaptionGeneratorBasic model. <br />
(default beam search @k is set to 5)








