import numpy as np
import pandas as pd

test_dataframe = pd.DataFrame({'float64': [1.0, 3.0, 4, -5, 6],
                    'float64_nulls': [1.0, -3.0, np.nan, np.inf, None],
                    'int64': [1, 2, 3, -5, 6],
                    'int64_nulls_const': [1, -2, np.nan, np.inf, None],
                    'int64_const': [1, 1, 1, 1, 1], 
                    'int64_nulls_const': [1, 1, np.nan, np.inf, None], 
                    'bool': [True, False, True, False, True],
                    'bool_null': [True, False, np.nan, np.inf, None],
                    'datetime64': [pd.Timestamp('20180310'), pd.Timestamp('20190310'), pd.Timestamp('20200310'), 
                                   pd.Timestamp('20210310'), pd.Timestamp('20220310')],
                     'datetime64_time': [pd.Timestamp('20180310 13:00:15'), pd.Timestamp('20190310 13:00:15'), 
                                         pd.Timestamp('20200310 13:00:15'), pd.Timestamp('20210310 12:00'), 
                                         pd.Timestamp('20220310 12:00')],
                    'datetime64_nulls': [pd.Timestamp('20180310'), pd.Timestamp('20190310'), np.nan, np.nan, None],
                    'object': ['foo','buzz', 'buzz','buzz', 'buzz'], 
                    'object_nulls': ['foo','buzz', np.nan, np.inf, None], 
                    'nans_inf_nones': [np.nan, np.inf, None, np.nan, None], 
                    'nans_nones': [np.nan, np.nan, None, np.nan, None],
                    'nones': [None, None, None, None, None]
                   })

def dataframe_metadata(data):

    if isinstance(data, pd.DataFrame):

        master = pd.DataFrame()
      
        for column in data.columns:
            datatypes = str(data.dtypes[column])

            n = data.shape[0]
            nulls = data[column].isna().sum()
            non_nulls = n - nulls
            proc_nulls = round((nulls/n)*100, 2)
            unique = len(data[column].unique())

            name_count = column+ '_count'
            val = data.groupby(column, dropna=False).size().reset_index(name=name_count)
            val.sort_values(name_count, ascending=False, inplace=True)
            val_list = val[column].to_list()

            max_elem = 20
            n_elem = len(val_list)

            data_list = []
            for index in range(0, min(n_elem, max_elem)):
                data_list.append(val_list[index])
            
            val_str = str(data_list)

            data_dict = {'name': [column], 'types': [datatypes], 
                        'total': [n], 'non_nulls': [non_nulls], 'nulls': [nulls], 'proc_nulls': [proc_nulls],
                        'unique': [unique],
                        'data': val_str
                        }
            row = pd.DataFrame.from_dict(data_dict)
            master = pd.concat([master, row])

        master = master.reset_index(drop=True)
        return(master)

    else:
        return(pd.DataFrame())