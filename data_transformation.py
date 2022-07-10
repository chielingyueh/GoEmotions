import argparse
import logging
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_emotion_ids(emotion_ids, emotions):
    emotion_ids_list = emotion_ids.split(",")

    emotion_names = []
    for idx in emotion_ids_list:
        emotion_names.append(emotions.iloc[int(idx)][0])

    return emotion_names


def ekman_emotions(emotion_list, ekman_mapping):
    ekman = []
    for emotion in emotion_list:
        if emotion != 'neutral':
            key = [k for k, v in ekman_mapping.items() if emotion in v]
            if key[0] not in ekman:
                ekman.append(key[0])
        else:
            ekman.append('neutral')
    return ekman


def ekman_ids(ekman_list, emotions_ekman):
    ekman = []
    for x in ekman_list:
        idx = np.where(emotions_ekman['emotion'] == x)[0][0]
        ekman.append(int(idx))
    return ekman


def one_hot_encoder(df):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0] * 7
        label_indices = df.iloc[i]["ekman_ids"]
        for index in label_indices:
            temp[index] = 1
        one_hot_encoding.append(temp)

    return pd.DataFrame(one_hot_encoding)


def transform_df(transform_config):
    train = pd.read_csv(transform_config.path_to_input_file, sep='\t', header=None, names=['text', 'emotion_ids', 'id'])
    emotions = pd.read_csv('./data/emotions.txt', header=None, names=['emotion'])

    train['emotions'] = train['emotion_ids'].apply(transform_emotion_ids, emotions=emotions)

    with open("./data/ekman_mapping.json", "r") as read_file:
        ekman_mapping = json.load(read_file)

    train['ekman_emotions'] = train['emotions'].apply(ekman_emotions, ekman_mapping=ekman_mapping)

    emotions_ekman = pd.read_csv('./data/ekman_emotions.txt', header=None, names=['emotion'])

    train['ekman_ids'] = train['ekman_emotions'].apply(ekman_ids, emotions_ekman=emotions_ekman)

    train_ohe_labels = one_hot_encoder(train)
    # valid_ohe_labels = one_hot_encoder(valid)
    # test_ohe_labels = one_hot_encoder(test)

    # print(train_ohe_labels.shape)
    # (43410, 28)

    train = pd.concat([train, train_ohe_labels], axis=1)
    # valid = pd.concat([valid, valid_ohe_labels], axis=1)
    # test = pd.concat([test, test_ohe_labels], axis=1)

    cols = [0, 1, 2, 3, 4, 5, 6]
    train['list'] = train[cols].values.tolist()

    new_df = train[['text', 'list']]
    # new_df = train[['text']]

    new_df = new_df.rename(columns={'text': 'comment_text'})

    new_df.to_csv(f'./data/{transform_config.dataset_type}_input.csv', index=False)

    logger.info(f'Created {transform_config.dataset_type} inputfile, see path: '
                f'"./data/{transform_config.dataset_type}_input.csv"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_input_file",
        default="./data/train.tsv",
        type=str,
        help="Path to file which needs to be transformed for input.",
    )

    parser.add_argument(
        "--dataset_type",
        default="train",
        type=str,
        help="Dataset type: train, dev, test.",
    )

    transform_config = parser.parse_args()
    transform_df(transform_config)

