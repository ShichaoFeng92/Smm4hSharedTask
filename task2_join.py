import numpy as np
import pandas as pd

reader1 = pd.read_csv('train_tweet.tsv', sep='\t', names=[
                      'tweet_id', 'user_id', 'key', 'tweet'])
reader2 = pd.read_csv('train_tweet_annotations.tsv', sep='\t', names=[
                      'key', 'feature1', 'feature2', 'class', 'AE', 'medication1', 'medication2'])
new = reader1.set_index('key').join(reader2.set_index('key'))
unique = new.drop_duplicates(subset=['tweet_id'])
reader1 = pd.read_csv('test_tweet.tsv', sep='\t', names=[
                      'tweet_id', 'user_id', 'key', 'tweet'])
reader2 = pd.read_csv('test_tweet_annotations.tsv', sep='\t', names=[
                      'key', 'feature1', 'feature2', 'class', 'AE', 'medication1', 'medication2'])
new = reader1.set_index('key').join(reader2.set_index('key'))
unique2 = new.drop_duplicates(subset=['tweet_id'])
unique_all = unique.append(unique2, ignore_index=True)
unique_all = unique_all[['tweet_id', 'tweet', 'class']]
for line_id, line in enumerate(unique_all['class'].isnull()):
    if line is True:
        unique_all['class'][line_id] = 0

for line_id, line in enumerate(unique_all['class']):
    if line is not 0:
        if line == 'ADR':
            unique_all['class'][line_id] = 1
        else:
            unique_all['class'][line_id] = 0

data = pd.read_csv('DRUG-AE.rel', sep='|', names=[
                   'tweet_id', 'tweet', 'med', 'feature1', 'feature2', 'AE', 'feature3', 'featur4'])

data = data[['tweet_id', 'tweet']]
data = data.drop_duplicates(subset=['tweet'])
label = [1 for line in data['tweet']]
data.insert(loc=2, column='class', value=label)

df = unique_all.append(data, ignore_index=False)
df.to_csv('additional_task2_en.csv', index=False)
