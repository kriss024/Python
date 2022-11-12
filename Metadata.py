import numpy as np
import pandas as pd

test_dataframe = pd.DataFrame({'float64': [1.0, 3.0, 4, -5, 6],
                    'float64_null': [1.0, -3.0, np.nan, np.inf, None],
                    'int64': [1, 2, 3, -5, 6],
                    'int64_null_const': [1, -2, np.nan, np.inf, None],
                    'int64_const': [1, 1, 1, 1, 1], 
                    'int64_null_const': [1, 1, np.nan, np.inf, None], 
                    'bool': [True, False, True, False, True],
                    'bool_null': [True, False, np.nan, np.inf, None],
                    'datetime64': [pd.Timestamp('20180310'), pd.Timestamp('20190310'), pd.Timestamp('20200310'), 
                                   pd.Timestamp('20210310'), pd.Timestamp('20220310')],
                     'datetime64_time': [pd.Timestamp('20180310 13:00:15'), pd.Timestamp('20190310 13:00:15'), 
                                         pd.Timestamp('20200310 13:00:15'), pd.Timestamp('20210310 12:00'), 
                                         pd.Timestamp('20220310 12:00')],
                    'datetime64_null': [pd.Timestamp('20180310'), pd.Timestamp('20190310'), np.nan, np.nan, None],
                    'object': ['foo','buzz', 'buzz','buzz', 'buzz'], 
                    'object_null': ['foo','buzz', np.nan, np.inf, None], 
                    'nans_inf_nones': [np.nan, np.inf, None, np.nan, None], 
                    'nans_nones': [np.nan, np.nan, None, np.nan, None],
                    'nones': [None, None, None, None, None]
                   })

def dataframe_metadata(data):

    if isinstance(data, pd.DataFrame):

        master = pd.DataFrame()
      
        for column in data.columns:
            datatype = str(data.dtypes[column])
            total = data.shape[0]
            null = data[column].isna().sum()
            non_null = total - null
            proc_null = round((null/total)*100, 2)
            unique = len(data[column].unique())

            name_count = column + '_count'
            val = data.groupby(column, dropna = False).size().reset_index(name = name_count)
            val.sort_values(name_count, ascending = False, inplace = True)
            val_list = val[column].to_list()

            max_elem = 20
            n_elem = len(val_list)
            min_elem = min(max_elem, n_elem)

            data_list = list()
            for index in range(0, min_elem):
                data_list.append(val_list[index])
            
            val_str = str(data_list)

            data_dict = {'name': [column], 'type': [datatype], 
                        'total': [total], 'non_null': [non_null], 'null': [null], 'proc_null': [proc_null],
                        'unique': [unique],
                        'data': val_str
                        }
                        
            row = pd.DataFrame.from_dict(data_dict)
            master = pd.concat([master, row])

        master = master.reset_index(drop = True)
        return(master)

    else:
        return None


def calculate_woe(data, column, default_column):

    if isinstance(data, pd.DataFrame):

        total_number = 'total_number'
        number_bad = 'number_bad'
        number_good = 'number_good'
        percent_total = 'percent_total'
        percent_bad = 'percent_bad'
        percent_good = 'percent_good'
        distribution_bad = 'distribution_bad'
        distribution_good = 'distribution_good'
        woe = 'woe'
        good_minus_bad = 'good_minus_bad'
        iv = 'iv'

        result = data.groupby(column, dropna=False)[default_column].aggregate(['count','sum'])
        result.columns = [total_number, number_bad]
        result.reset_index(inplace=True)
        result[number_good] = result[total_number] - result[number_bad]
        sum_ = result.agg('sum')
        result[percent_total] = (result[total_number] / sum_[total_number]) * 100
        result[percent_bad] = (result[number_bad] / sum_[total_number]) * 100
        result[percent_good] = (result[number_good] / sum_[total_number]) * 100
        result[distribution_bad] = result[number_bad] / sum_[number_bad]
        result[distribution_good] = result[number_good] / sum_[number_good]
        result[woe] = np.log(result[distribution_good] / result[distribution_bad])
        result[good_minus_bad] = result[distribution_good] - result[distribution_bad]
        result[iv] = result[good_minus_bad] * result[woe]
        iv_calculated = result[iv].agg('sum')

        return result, iv_calculated

    else:
        return None