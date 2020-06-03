## Creating virtualenv(python3.6) && install dependencies

```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m deeppavlov install ner_ontonotes_bert_mult
```