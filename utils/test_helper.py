import pandas as pd
from io import BytesIO
import numpy as np


from helper import load_data, transform_dataset

def test_load_data():
    """
    Test the load data function
    Create a file in memory and test loading
    """

    # Excel
    df = pd.DataFrame({'Data': [1,2,3,4]})
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    xlsx_data = load_data(output, 'Excel File')
    pd.testing.assert_frame_equal(xlsx_data, df)

def test_transform_dataset():
    """
    Test the transform dataset function
    Test if the transformation is done correctly
    """

    df = pd.DataFrame(np.array([[1, 2, 'm'], [4, 5, 'w'], [7, 8, 'm']]), columns=['a', 'b', 'c'])
    df_t = transform_dataset(df, ['c'],['a','b'])
    assert df_t['c'].dtype == np.dtype('int')
