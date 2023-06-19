import re
from json import dump as jdump
from json import load as jload
from os import chdir

tag_list = ["ADJ",
"ADP",
".",
"ADV",
"AUX",
"SYM",
"INTJ",
"CONJ",
"X",
"NOUN",
"DET",
"PROPN",
"NUM",
"VERB",
"PRT",
"PRON",
"SCONJ"]
target_dir = r"C:\Users\drews\PycharmProjects\POS_AAVE_495R"
print(chdir(target_dir))
def dict_lst_to_json(data_list:list[dict], file_path:str)-> None:
    with open(file_path, 'w') as file:
        jdump(data_list, file)
    print(f"Data successfully written to {file_path}.")

def read_dicts_from_json(file_path):
    with open(file_path, 'r') as file:
        data = jload(file)
    return data

with open(r"C:\Users\drews\Downloads\data_2_b_processedtxt.txt", mode='r', encoding='utf8') as inf:
    txt = inf.readlines()  # 2374 tweets; where can we get more PoS labeled data?
## Sanity check
# print(txt[:10])

n = 0
all_tweets_list = []
temp_wds = []
for line in txt:
    if line == "\n":
        all_tweets_list.append({})
        all_tweets_list[n]["ID"] = n
        all_tweets_list[n]["Tweet"] = temp_wds
        temp_wds = []
        n += 1
    else:
        temp_wds.append(line)
        # print(temp_wds)

# sanity check
# [print(all_tweets_list[i]) for i in range(len(all_tweets_list))]

## Line example:
# NOUN 'd-1-9| w=coach pref1=c suf3=ach <w=.. <suf3=.. <<w=tolls >w=comin >suf3=min >>w=outta
PoS_set = set()
errors = 0
for tweet in all_tweets_list:
    wds = []
    PoSes = []
    for line in tweet["Tweet"]:
        PoS = re.findall("^(\S+)\s", line, re.I)  # Get the part of speech
        wd = re.findall("\sw=(\S+)\s", line, re.I)  # Get the corresponding word
        wds.append(wd[0])
        PoSes.append(tag_list.index(PoS[0]))
    for part in PoSes:
        PoS_set.add(part)

    # zipped_lists = zip(PoSes, wds)
    tweet["Tweet"] = " ".join(wds)
    tweet["Words"] = wds
    tweet["Tags"] = PoSes

all_tweets_list.pop(0)

# from datasets import Dataset

# def create_dataset_from_dict(data_dict):
#     dataset = Dataset.from_dict(data_dict)
#     return dataset


## Sanity check
# for i in range(5):
#     print(all_tweets_list[i])

## Write out to .txt file
# dict_lst_to_json(all_tweets_list, 'tweets.json')

## Read in
# x = read_dicts_from_json('tweets.json')


train_count = (round(len(all_tweets_list)*.75))
validate_count = (round(len(all_tweets_list)*.15))

train_tweets = all_tweets_list [:train_count]  # ~1780
validate_tweets = all_tweets_list[train_count:train_count + validate_count]  # ~356
test_tweets = all_tweets_list[train_count+validate_count:]  # ~258
print(train_tweets[0]["Tags"])
out_list = [("train", train_tweets), ("validate", validate_tweets), ("test", test_tweets)]

# while len(out_list["validate"]) < len(out_list["train"]):
#     out_list["validate"].append({})
# while len(out_list["test"])< len(out_list["train"]):
#     out_list["test"].append({})
# print(len(out_list["train"]), len(out_list["validate"]), len(out_list["test"]))

# x = create_dataset_from_dict(sub_dict)
# print(type(x))
for i, j in out_list:
    dict_lst_to_json(j, f'{i}_tweets.json')



