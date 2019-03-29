Neural Character-level Part-of-speech (POS) tagger of Slovene language. Neural model is based on character level language model proposed by by Kim et al. in Character-Aware neural language models (2015).

Our POS tagger has been presented in thesis ***Part of speech tagging of slovene language using deep neural networks***. Thesis is accessible at [University of Ljubljana](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=105266&lang=eng)


### Abstract
*The thesis deals with part of speech tagging of Slovene language. Part of speech tagging is a process of matching sentences in natural language with a sequence of suitable tags, which contain information about parts of speech and morphological properties of words. Our solution uses character-level representation of words, which is different from typical solutions, which process input sentences as sequences of words. Our part of speech tagger is implemented using convolutional and recurrent neural networks. Unlike common approaches that address this problem as multi-class classification, our solution proposes a multi-label classification approach. In order to improve our results we implement an ensemble of three part of speech taggers. When comparing our solution with existing ones, we find that the proposed solution achieves the best results.*


## Training corpus
Our model has been trained on SSJ500k training corpus. This corpus is available at clarin.si: [Training corpus SSJ500k 2.2](https://www.clarin.si/repository/xmlui/handle/11356/1210) Our model has been trained using ssj500k-en.TEI.


## Trained model
Due to file size limitations of GitHub, trained model (model_weights.h5) is not available in this repository. It can be acquired at clarin.si repository: [Part-of-speech tagger of Slovene language](https://www.clarin.si/repository/xmlui/handle/11356/1211)


## Examples

### Tagging using pretrained model

Tagger loads Keras model configuration from `model.json` and model weights from `model_weights.h5`.

#### Tagging sentences in XML/TEI format ####

```
python tag.py input.xml output.xml
```

#### Tagging sentences in plain text files ####

Tagging of text files requires [Obeliks4J](https://github.com/clarinsi/Obeliks4J) for segmentation and tokenisation. Path to Obeliks4J directory must be passed through `--obelikspath` parameter.
```
python tag.py input.txt output.xml --obelikspath /home/user/Obeliks4J/
```

#### Optional parameters ####
```
-s, --slo	Tagger predicts slovene tags.
-f, --force	Overwrite existing output file.
```


### Training models

Training new models requires dataset in XML\TEI format. Output of training process consists of files `model.json` which contains Keras model configuration, `model_weights.h5` containing weights of neural network and `charset` which contains a list of all character occuring in provided dataset. These three files are are used to make predictions with tag.py.

Number of training iterations (epochs) can be passed with -n parameter. Default number of epochs is 20.

#### Training a model for 10 iterations ####
```
python train.py ssj500k.xml outputdir -n 10
```

#### Optional parameters ####
```
-n, --nepoch	Number of training iterations.
-s, --slo	Indicates that tags in training set are in slovene language.
-b, --beginning	Can be used to train model on subset of input training set. Starting (inclusive) index of subset.
-e, --end	Can be used to train model on subset of input training set. Final (exclusive) index of subset.
```
