import transformers, evaluate

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("DrewG/AAVE_PoS-Tagger")

model = AutoModelForTokenClassification.from_pretrained("DrewG/AAVE_PoS-Tagger")

precision_metric = evaluate.load("accuracy")

# https://bitbucket.org/soegaard/aave-pos16/src/master/data/test/self/testwire_cmuc.vw
# Use this^^ and other files in the directory