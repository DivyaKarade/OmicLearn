import sys
import numpy as np
import pandas as pd
sys.path.append('..')
from io import BytesIO
from utils.helper import load_data, transform_dataset, normalize_dataset

def test_load_data():
    """
    Test the load data function
    Create a file in memory and test loading
    """

    # Excel
    df = pd.DataFrame({'Data': [1, 2, 3, 4]})
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    xlsx_data = load_data(output, 'Excel File')
    pd.testing.assert_frame_equal(xlsx_data, df)

    # csv
    df = pd.DataFrame({'A': [1, 1], 'B': [0, 0]})
    csv_data = load_data('test_csv_c.csv', 'Comma (,)')
    print(csv_data)
    pd.testing.assert_frame_equal(csv_data, df)

    csv_data = load_data('test_csv_sc.csv', 'Semicolon (;)')
    print(csv_data)
    pd.testing.assert_frame_equal(csv_data, df)

def test_transform_dataset():
    """
    Test the transform dataset function
    Test if the transformation is done correctly
    """

    df = pd.DataFrame(np.array([[1, 2, 'm', '+'], [4, 5, 'w', '-'], [7, 8, 'm', '-']]), columns=['a', 'b', 'c', 'd'])
    df_t = transform_dataset(df, ['c'], ['a', 'b'])
    assert df_t['c'].dtype == np.dtype('int')

    df_t = transform_dataset(df, ['c', 'd'], ['a', 'b'])
    assert df_t['c'].dtype == np.dtype('int')
    assert df_t['d'].dtype == np.dtype('int')

    df_t = transform_dataset(df, [], ['a', 'b'])

    for column in df_t.columns:
        assert df_t[column].dtype == np.dtype('float')

def test_normalize_dataset():
    """
    Tests the normalization
    Calls all the Normalization Methods
    """

    df = pd.DataFrame({'Data': [1, 2, 3, 4]})

    for normalization in ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']:
        if normalization == 'PowerTransformer':
            normalization_detail = 'box-cox'
            n_quantiles = None
        elif normalization == 'QuantileTransformer':
            normalization_detail = 'uniform'
            n_quantiles = 1000
        else:
            normalization_detail = ''
            n_quantiles = None
        normalize_dataset(df, normalization, normalization_detail, n_quantiles, 23)