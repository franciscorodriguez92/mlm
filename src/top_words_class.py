#%%
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.datasets_exist import TextCleaner
#%%
language = "en"
from nltk.corpus import stopwords
#stop = stopwords.words('english')
#stop = stopwords.words('spanish')
if language == "es":
    stop = stopwords.words('spanish')
elif language == "both":
    stop = stopwords.words('spanish')
    stop = stop + stopwords.words('english')
else:
    stop = stopwords.words('english')
#%%
file_exist_2021 = "../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2022_training.tsv"
df_file_exist_2021= pd.read_table(file_exist_2021, sep="\t")
file_exist_2022 = "../data/input/EXIST 2022 Dataset/test/test_EXIST2022_labeled.tsv"
df_file_exist_2022= pd.read_table(file_exist_2022, sep="\t")
df_file_exist_2022= df_file_exist_2022.rename(columns={"text_case": "test_case"})
df_file_exist_2022.drop('status_id', axis=1, inplace=True)
df_file_exist_all = pd.concat([df_file_exist_2021, df_file_exist_2022])
df_file_exist_all = df_file_exist_all[
    df_file_exist_all['task2']!="desacuerdo"]
df_file_exist_all['task2']=df_file_exist_all['task2'].map({'non-sexist' : 0, 'ideological-inequality': 1, 'stereotyping-dominance': 2, 
				'objectification': 3, 'sexual-violence': 4, 'misogyny-non-sexual-violence': 5})

preprocessor = TextCleaner(filter_users=True, filter_hashtags=False, 
                           filter_urls=True, convert_hastags=False, lowercase=False, 
                           replace_exclamation=False, replace_interrogation=False, 
                           remove_accents=False, remove_punctuation=False)
df_file_exist_all['text'] = df_file_exist_all['text'].apply(lambda row: preprocessor(row))
df_file_exist_all['text'] = df_file_exist_all['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


#%%
if language != "both":
    df_file_exist_all = df_file_exist_all[
        df_file_exist_all['language']==language]
#%%
# Skip over dimensionality reduction, replace cluster model with classifier,
# and reduce frequent words while we are at it.
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
#clf = RandomForestClassifier()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

#%%
# Create a fully supervised BERTopic instance
topic_model= BERTopic(
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model,
        language="multilingual"
)

#%%
topics, probs = topic_model.fit_transform(df_file_exist_all['text'].values, 
                                          y=df_file_exist_all['task2'].values)
# %%
topic_model.get_topic_info()

#%%
df_file_exist_all['task2'].value_counts()
#%%
topic_model.get_topic(0)

# %%
topic_model.get_topic(1)
#%%
topic_model.get_topic(2)
#%%
topic_model.get_topic(3)
#%%
topic_model.get_topic(4)
#%%
topic_model.get_topic(5)
# %%
#topic_model.visualize_topics()

# %%
