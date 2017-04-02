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

## TODO
- Beam Search
- Schedule Sampling
