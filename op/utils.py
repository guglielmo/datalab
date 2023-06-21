from collections import Counter
import json
import string
import re
import random
import zipfile

import boto3
from bs4 import BeautifulSoup
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import _hash_file
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer



def get_act_tags(act_id: int, tags_df, tagging_df) -> list:
    """Return all tags for the act having given id, as triplets:
      0 - the tag id,
      1 - the tag text
      2 - the tag type (teseo|geoteseo|op_geo)
    """
    tags_list = list(tagging_df[tagging_df.act_id==act_id].tags_ids)
    if len(tags_list):
        tags_list = tags_list[0].split(':')[1:]
    return [
        (
            tags_df[tags_df.id==int(t)].iloc[0]['id'], 
            tags_df[tags_df.id==int(t)].iloc[0]['name'], 
            tags_df[tags_df.id==int(t)].iloc[0]['type']
        ) for t in tags_list
    ]


def get_act_tags_ids(act_id: int) -> list:
    """Return all tags ids, for non-geo tags"""
    return [t[0] for t in get_act_tags(act_id) if 'geo' not in t[2]]


def extract_text_from_html(html: string) -> string:
    """Extract text from act's HTML content, 
       removing names of MPs that signed the act
    """
    soup = BeautifulSoup(html, 'html.parser')
    stripped = soup.get_text().replace("\n", " ").replace("  ", " ").strip(' ')
    text = re.sub(r'(\([\d\-]*\)) «(.*)»\.', '', stripped).strip()
    return text


def nltk_process(text: string, remove_words: list = None) -> list:
    """Process a text with NLTK doing the following:
    - split into words,
    - convert to lower case
    - remove punctuation
    - filter out stop words
    
    :param: text - the text to be processed by nltk tools
    :param: remove_words - the list of common words to remove (if not None)
    :return: a list of stemmed words
    """
    
    if remove_words is None:
        remove_words = []
    else:
        remove_words = [w.lower() for w in remove_words]
        
    # need this for italian texts
    text = text.replace("'", " ")
    
    # split into words
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens if w.lower() not in remove_words]

    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('italian')) | {'n'}
    words = [w for w in words if not w in stop_words]

    # stemming of words
    # stemmer = SnowballStemmer("italian")
    # stemmed = [stemmer.stem(word) for word in words]

    return words


def preprocess_and_save_docs(
    source_zip: string, dest_name: string, 
    tags_df, tagging_df,
    n_docs: int = 100,
    cache_path: string = './datasets',
    aws_bucket: string = 'opp-datasets',
    aws_region: string = 'eu-central-1',
    aws_access_key_id: string = None,
    aws_secret_access_key: string = None
) -> tuple:
    """Pre-process ``n_docs`` random documents out of the ones contained in `source_zip` file, 
    pre-cleaning them for serialization onto disk, along with labels (tags).
    
    Data are serialized in the ``dest_npz`` file, with the `.npz` compressed format.
    A vocabulary is serialized in a json file, with thename extracted from dest_npz.

    For each one of the n_docs documents in the zipped file:
    - html is parsed and text content is extracted (beautifulsoup)
    - both the data list and the vocab Counter are updated
    
    :param: source_zip - the zipfile comlpete path
    :param: dest_name - the name of the file, no path, no extension
    :param: tags_df - the pandas DataFrame containing the tags, id, name, type
    :param: tagging_df - the pandas DataFrame containing the association between the acts and the tags
    :param: n_docs: - the number of documents to extract from the zip file
    :param: cache_path - the path where npz and json file should be stored before being uploaded to S3
    :param: aws_bucket - aws s3 bucket name
    :param: aws_region - aws region for the bucket
    :param: aws_access_key_id - aws key_id with permission to upload to S3
    :param: aws_secret_access_key - aws secret key for use with permission to upload to S3
    
    :return: a tuple with the MD5 hashes of the npz and json files, respectively
    
    The following data are persisted in ``dest_npz``, using the ``.npz`` format:
      ids:    list of original openparlamento ID (to refer to the original ACT)
      data:   list of original texts (pre-cleaned)
      labels: list of the assigned labels 
              each label is a triple: (ID, name, type)
      vocab:  the complete vocabulary, with occurence counts for each word
    """
    data = []
    vocab = Counter()

    assert(aws_access_key_id is not None and aws_secret_access_key is not None)
    
    with open(source_zip, 'rb') as tz:
        z = zipfile.ZipFile(tz)
        filelist = random.choices(z.filelist, k=n_docs)
        for fl in tqdm(filelist):
            zf = z.open(fl.filename)
            original_html = zf.read()
            text_content = extract_text_from_html(original_html)
            words = nltk_process(text_content)            
            zf.close()
            id = int(fl.filename.split('_')[0])
            data_f = {
                'id': id,
                'text_content': text_content,
                'tags': get_act_tags(id, tags_df=tags_df, tagging_df=tagging_df),
            }
            data.append(data_f)
            vocab.update(words)

    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    dest_npz = f"{cache_path}/{dest_name}.npz"
    dest_vocab = f"{cache_path}/{dest_name}_vocab.json"

    print(f"Saving data into {dest_npz}")
    np.savez_compressed(
        dest_npz, 
        ids=[doc['id'] for doc in data], 
        texts=[doc['text_content'] for doc in data], 
        labels=[doc['tags'] for doc in data], 
    )
    print(f"Uploading to s3://{aws_bucket}/{dest_name}.npz")
    s3.upload_file(
        dest_npz, aws_bucket, f"{dest_name}.npz",
        ExtraArgs={'ACL': 'public-read'}
    )
    
    print(f"Saving vocab into {dest_vocab}")
    with open(dest_vocab, 'w') as outfile:
        json.dump(vocab, outfile)
    print(f"Uploading to s3://{aws_bucket}/{dest_name}_vocab.json")
    s3.upload_file(
        dest_vocab, aws_bucket, f"{dest_name}_vocab.json",
        ExtraArgs={'ACL': 'public-read'}
    )

    return (
        (f"https://{aws_bucket}.s3.{aws_region}.amazonaws.com/{dest_name}.npz", 
         _hash_file(dest_npz, algorithm='md5')), 
        (f"https://{aws_bucket}.s3.{aws_region}.amazonaws.com/{dest_name}_vocab.json", 
         _hash_file(dest_vocab, algorithm='md5'))
    )

def vectorize(
    texts_raw: list, labels_raw: list, data_type: string,
    include_tag_types: list = None,
    remove_words: list = None,
    num_words: int = 10000,
    text_to_matrix_mode: string = 'binary',
    all_labels: list = None
) -> tuple:
    """Transform raw values for tests and labels into Tensors.
    
    Data are transformed into a **matrix**, using 
        ``keras.preprocessing.text.Tokenizer.text_tom_matrix``,
    
    Labels are transformed into an Array, using
        ``sklearn.preprocessing.MultiLabelBinarizer.fit_transform``.
    
    :param: texts_raw - a numpy array of text contents
    :param: labels_raw - a numpy array of lists of tuples (id, name, type)
    :param: data_type - whether train or test
    :param: include_tag_type - tags containing this string are filtered out
    :param: remove_words - list of words to remove, as non-interesting
    :param: num_words - maximum number of words (to fix the size of the matrix)
    :param: text_to_matrix_mode - mode to use in text_to_matrix (binary, freq, itfdf,)
    
    :return: - a tuple containing the texts and labels tensors, and the original, untrasformed labels (ids)
    
    Note: the untransformed labels are returned in order to be able to decode 
    results found from the model from tensor form back to ids.
    
    Note that by default only tags not containing 'geo' in them are passed.
    This behaviour can be changed using the ``filter_out_tag_type`` parameter.
    
    """
    if remove_words is None:
        remove_words = []
    else:
        remove_words = [w.lower() for w in remove_words]

    texts = []
    for x in tqdm(texts_raw, desc=f"{data_type} texts"):
        texts.append(" ".join(nltk_process(x, remove_words=remove_words)))
    
    if include_tag_types is None:
        include_tag_types = ['teseo', 'op_geo', 'geoteseo', 'user']

    labels = []
    for y in tqdm(labels_raw, desc=f"{data_type} labels"):
        labels.append(
            [x[0] for x in y if x[2] in include_tag_types]
        )

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    x = tokenizer.texts_to_matrix(texts, mode=text_to_matrix_mode)
    
    one_hot = MultiLabelBinarizer()
    y = one_hot.fit_transform(labels)

    return x, y, labels


def plot_history(history, title="History"):

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)

    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 4))
    fig.suptitle(title, fontsize=16)

    axs[0].plot(epochs, loss_values, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
    axs[0].set_title("Training and validation loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(epochs, acc, 'bo', label="Training accuracy")
    axs[1].plot(epochs, val_acc, 'b', label="Validation accuracy")
    axs[1].set_title("Training and validation accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid()

    plt.show()