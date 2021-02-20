import torch
import random
import numpy as np
import copy
import pandas as pd

def column_name_converter(string):
    return string.lower().replace("\n", "_").replace(" ", "_")

def change_column_names(df):
    column_name0 = list(df.columns)
    column_name1 = [column_name_converter(column) for column in column_name0]
    column_name2 = list(df.iloc[0])
    assert len(column_name1) == len(column_name2)
    prefix = ''
    for i, name in enumerate(column_name2):
        if isinstance(name, str):
            if prefix != column_name1[i] and 'unnamed:' not in column_name1[i]:
                prefix = column_name1[i]
            new_name = prefix + "_" + column_name_converter(name)
            column_name2[i] = new_name
        else:
            column_name2[i] = None

    column_name3 = []
    for i, name in enumerate(column_name2):
        if name is not None:
            column_name3.append(name)
        else:
            column_name3.append(column_name1[i])

    df = df.rename(columns={column_name0[i]: column_name3[i] for i in range(len(column_name1))})
    df = df.drop(0)
    df = df.sort_values(by='date').reset_index()
    del df['index']
    return df


def handle_neural_index(df):
    key = 'neural_index'
    if key in list(df.keys()):
        for i in df.index:
            df.loc[i, key] = 0.0 if df.loc[i, key] == 'down' else 1.0
    return df


def rule_closeup(df):
    """
        returns a DataFrame column that is true (1) if the current dates
        close_price is higher than the close_price of the previous date
    """
    assert 'close_price' in list(df.columns)
    assert 'date' in list(df.columns)

    closeup = [None]
    for i in range(1, len(df.index)):
        previous_close_price = df.iloc[i - 1]['close_price']
        current_close_price = df.iloc[i]['close_price']
        closeup.append(int(current_close_price > previous_close_price))
    assert len(closeup) == len(df.index)
    return closeup


def df_to_src_tgt(config=None, df=None):
    src_keys = list(df.keys())

    tgt_keys = [config['tgt_key']]
    for key in tgt_keys + config['forbidden_keys']:
        if key in src_keys:
            src_keys.remove(key)
    src_pos_dict = {key: i for i, key in enumerate(src_keys)}
    src_data = df[src_keys].to_numpy()
    tgt_data = df[tgt_keys].to_numpy()
    return src_data, tgt_data, src_keys, tgt_keys, src_pos_dict


def create_input(config=None, src_data=None, src_keys=None, tgt_data=None, src_pos_dict=None):
    assert src_data.shape[0] == tgt_data.shape[0]

    n_datapoints = src_data.shape[0]
    sequence_length = config['sequence_length']
    data = []
    close_price_idx = src_pos_dict['close_price']
    for i in range(sequence_length, n_datapoints, sequence_length):
        src = src_data[i - sequence_length:i]
        tgt = tgt_data[i]
        assert bool(tgt[0]) == (src[-1, close_price_idx] < src_data[i, close_price_idx])
        data.append((src, tgt))
    return data

def batch_generator(bs=8, data=None):
    assert data is not None
    random.shuffle(data)
    idx = bs
    while idx < len(data) + 1:
        batch_list = copy.deepcopy(data[idx - bs: idx])

        src, tgt = [], []
        for i in range(bs):
            src_tgt_tup = batch_list[i]
            src.append(src_tgt_tup[0])
            tgt.append(src_tgt_tup[1])
        src = np.array(src)
        tgt = np.array(tgt)
        src = src.astype(float)
        tgt = tgt.astype(float)
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)

        idx += bs
        yield (src.cuda(), tgt.cuda())

def get_data(config=None):
    assert config is not None
    df = pd.read_excel('res/Euro.xls')
    df = change_column_names(df)
    df = handle_neural_index(df)
    df = pd.concat([df, pd.DataFrame(data=rule_closeup(df), columns=['rule_closeup'])], axis=1)
    src_data, tgt_data, src_keys, tgt_keys, src_pos_dict = df_to_src_tgt(config=config, df=df)
    data = create_input(config=config, src_data=src_data, tgt_data=tgt_data, src_pos_dict=src_pos_dict)
    return data

def split_data(data):
    n_datapoints = len(data)
    return data[:int(n_datapoints * 0.9)], data[int(n_datapoints * 0.9):]
