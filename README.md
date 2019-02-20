## Examples

### Tagging using pretrained model

#### Tagging sentences in XML/TEI format ####

```
python tag.py input.xml output.xml
```

#### Tagging sentences in plain text files ####

Tagging of text files requires Obeliks4J (https://github.com/clarinsi/Obeliks4J) for segmentation and tokenisation.
```
python tag.py input.txt output.xml --obelikspath /home/user/Obeliks4J/
```

Optional parameters
```
-s, --slo	Tagger predicts slovene tags.
-f, --force	Overwrite existing output file.
```
