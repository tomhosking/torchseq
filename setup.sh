mkdir ./data/
curl https://nlp.stanford.edu/data/glove.6B.zip -o ./data/glove.6B.zip -L -C -
mkdir ./data/squad
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/train-v1.1.json -o ./data/squad/train-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/dev-v1.1.json -o ./data/squad/dev-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/test-v1.1.json -o ./data/squad/test-v1.1.json -L -C -
unzip ./data/glove.6B.zip -d ./data/glove.6B/
python -m spacy download en
rm ./data/glove.6B.zip
