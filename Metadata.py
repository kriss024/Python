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

catCol = np.array(list('ABCD'))[np.random.randint(4, size=100)]
catCol2 = pd.Categorical(catCol)
numCol = np.random.random(size=100)
intCol = np.random.randint(5, size=100)
intCol2 = np.random.randint(30, size=100)
datCol = pd.date_range(
    '2018-01-01', '2018-01-31')[np.random.randint(31, size=100)]
boolCol = np.array(list([True, False]))[np.random.randint(2, size=100)]
constCol = np.ones(100)

test_dataframe2 = pd.DataFrame({
    'catCol': catCol,
    'catCol2': catCol2,
    'intCol': intCol,
    'intCol2': intCol2,
    'numCol': numCol,
    'datCol': datCol,
    'boolCol': boolCol,
    'constCol_longlonglonglonglonglongName': constCol
})

def remove_all_by_values(list_obj, values):
    list_obj_rem = list_obj.copy()
    for value in set(values):
        while value in list_obj_rem:
            list_obj_rem.remove(value)
    return list_obj_rem

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


def calculate_woe_iv(dataset, feature, target):

    if isinstance(dataset, pd.DataFrame):

        lst = []

        values = dataset[feature].unique()
        for count, value in enumerate(values):
            lst.append({
                'Name' : feature,
                'Value': value,
                'Total': dataset[dataset[feature] == value].shape[0],
                'Good': dataset[(dataset[feature] == value) & (dataset[target] == 0)].shape[0],
                'Bad': dataset[(dataset[feature] == value) & (dataset[target] == 1)].shape[0]
            })
        
        dset = pd.DataFrame(lst)
        dset['Share_Total'] = dset['Total'] / dset['Total'].sum()
        dset['Share_Good'] = dset['Good'] / dset['Total']
        dset['Share_Bad'] = dset['Bad'] / dset['Total']
        dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
        dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
        dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
        dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
        iv = dset['IV'].sum()
        
        dset = dset.sort_values(by='WoE')
        
        return dset, iv

    else:
            return None
