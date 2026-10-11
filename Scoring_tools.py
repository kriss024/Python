'''
Created on 11-10-2026

@author   : Krzysztof Bruszewski
@email    : krzysztof.bruszewski@gmail.com
@file     : Scoring_tools.py
'''

import math
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from scipy import stats
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.cluster import KMeans
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Any

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def calculate_woe(df: pd.DataFrame,
                  cat_variable: str,
                  target_variable: str
                  ) -> pd.DataFrame:

    # Create a DataFrame to store the counts and target variable distribution
    data = pd.DataFrame({'Category': df[cat_variable], 'Target': df[target_variable]})
    data['Events'] = np.where(data['Target'] == 1, 1, 0)
    data['Non-Events'] = np.where(data['Target'] != 1, 1, 0)

    # Calculate the total number of events and non-events
    total_events = data['Events'].sum()
    total_non_events = data['Non-Events'].sum()
    total_count = total_events + total_non_events

    # Calculate the count and distribution of events and non-events for each category
    category_counts = data.groupby('Category', as_index = False, dropna = False).agg({'Events': 'sum', 'Non-Events': 'sum'})
    category_counts['Count'] = category_counts['Events'] + category_counts['Non-Events']
    category_counts['Bad_Rate'] = category_counts['Events'] / category_counts['Count']
    category_counts['Column_Name'] = cat_variable
    category_counts = category_counts[['Column_Name', 'Category', 'Count', 'Events', 'Bad_Rate', 'Non-Events']]
    category_counts['Events_Adjusted'] = np.where(category_counts['Events'] > 0, category_counts['Events'], 0.01)
    category_counts['Non-Events_Adjusted'] = np.where(category_counts['Non-Events'] > 0, category_counts['Non-Events'], 0.01)

    # Calculate the WoE and Information Value (IV) for each category
    category_counts['Event_Distribution'] = category_counts['Events_Adjusted'] / total_events
    category_counts['Non-Event_Distribution'] = category_counts['Non-Events_Adjusted'] / total_non_events
    category_counts['WoE'] = np.log(category_counts['Non-Event_Distribution'] / category_counts['Event_Distribution'])
    category_counts['IV'] = (category_counts['Non-Event_Distribution'] - category_counts['Event_Distribution']) * category_counts['WoE']
    category_counts = category_counts.drop(['Events_Adjusted', 'Non-Events_Adjusted'], axis=1)

    # Sort the DataFrame by the category for better visualization
    category_counts = category_counts.sort_values(by='Bad_Rate', ascending = False)

    # Calculate the overall IV
    overall_iv = category_counts['IV'].sum()

    return category_counts.reset_index(drop=True)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def woe_table_to_dict(df: pd.DataFrame) -> dict:
    woe_dict = {}
    for index, row in df.iterrows():
        woe_dict[row['Category']] = row['WoE']

    return woe_dict

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def rules_table_to_sql(df: pd.DataFrame,
                       column: str = 'Class',
                       else_val: Any = -1,
                       new_line_char: str = '\t'
                       ) -> str:
    sql_list = []
    sql_list.append('CASE')

    for index, row in df.iterrows():
        output = row[column]
        if not isinstance(output, str):
            output = str(output)
        sql_case = f"WHEN {row['Attribute']} THEN {output}"
        sql_list.append(sql_case)

    sql_list.append(f'ELSE {else_val}')
    classing_name = df['ClassingName'].iloc[0]
    sql_list.append(f'END AS {classing_name}')

    return new_line_char.join(sql_list)

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def auc_gini_statsmodel(df: pd.DataFrame,
                        indep_vars: list,
                        target_name: str,
                        model
                        )-> (float, float):

    y = df[target_name].values
    x = df.loc[:, indep_vars]
    X = sm.add_constant(x, has_constant = "add")
    preds = model.predict(X)

    # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y, preds)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    gini = 2.0*roc_auc-1.0

    return roc_auc, gini

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def auc_gini_pd(df: pd.DataFrame,
                   pd_name: str,
                   target_name: str
                  )-> (float, float):

    y = df[target_name].values
    y_score = df[pd_name].values

    # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y, y_score)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    gini = 2.0*roc_auc-1.0

    return roc_auc, gini

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def correlation_matrix_plot(df: pd.DataFrame,
                            indep_vars: list,
                            method: str = 'spearman'
                           ) -> None:
    corr_matrix = df[indep_vars].corr(method = method)
    corr_matrix = corr_matrix.round(2)

    cmap = sns.diverging_palette(500, 10, as_cmap = True)
    ans = sns.heatmap(corr_matrix,
                      linewidths = 1,
                      center = 0,
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      cmap = cmap,
                      annot_kws={"size": 7.5})

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def vcramer_corr_matrix_plot(df: pd.DataFrame,
                            indep_vars: list
                           ) -> None:

    cols_size = len(indep_vars)
    content = np.ones((cols_size, cols_size), dtype = float)
    vcramer_corr = pd.DataFrame(content, index = indep_vars, columns = indep_vars)

    all_combinations = list(itertools.combinations(indep_vars, 2))
    for pair in all_combinations:
        col1 = pair[0]
        col2 = pair[1]

        # Create a contingency table from the DataFrame
        contingency_table = pd.crosstab(df[col1], df[col2])

        # Calculate Chi-squared statistic
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate normalization factors
        n = contingency_table.sum().sum()  # Total sample size
        min_dim = min(contingency_table.shape) - 1

        # Cramer's V calculation
        v = np.sqrt(chi2 / (n * min_dim))
        vcramer_corr.loc[col1, col2] = v
        vcramer_corr.loc[col2, col1] = v

    corr_matrix = vcramer_corr.round(2)

    cmap = sns.diverging_palette(500, 10, as_cmap = True)
    ans = sns.heatmap(corr_matrix,
                      linewidths = 1,
                      center = 0,
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      cmap = cmap,
                      annot_kws={"size": 7.5})

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def calculate_ks(y_real, y_proba):

    # Create the empty DataFrame
    column_names = ['real', 'proba']
    df = pd.DataFrame(columns=column_names)
    df['real'] = y_real
    df['proba'] = y_proba

    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, pvalue = stats.ks_2samp(class0['proba'], class1['proba'])

    return ks_statistic, pvalue

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def psi_dataframe(expected: pd.Series,
                  actual: pd.Series,
                  buckets: int = 10
                  ) -> pd.DataFrame:

    expected = expected.rename('initial')
    actual = actual.rename('actual')
    points = list(np.arange(1, buckets, 1)/buckets)
    breakpoints = expected.quantile(points, interpolation='midpoint')
    breakpoints = breakpoints.values
    breakpoints = np.append(breakpoints, (-float("inf")))
    breakpoints = np.append(breakpoints, float("inf"))
    breakpoints = np.unique(breakpoints)
    breakpoints = np.sort(breakpoints)

    labels = list(range(1, breakpoints.size))

    expected_name = expected.name
    expected_df = pd.DataFrame(expected)
    expected_df['category'] = pd.cut(expected_df[expected_name], bins=breakpoints, labels=labels)
    expected_df['bucket'] = expected_df['category'].astype(float).fillna(0)
    expected_df['bucket'] = expected_df['bucket'].astype(int)
    expected_agg = expected_df.groupby('bucket', as_index=False).size()
    expected_agg.columns = ['bucket', 'expected']
    expected_sum = expected_agg['expected'].sum()
    expected_agg['expected_distribution'] = expected_agg['expected']/expected_sum

    actual_name = actual.name
    actual_df = pd.DataFrame(actual)
    actual_df['category'] = pd.cut(actual_df[actual_name], bins=breakpoints, labels=labels)
    actual_df['bucket'] = actual_df['category'].astype(float).fillna(0)
    actual_df['bucket'] = actual_df['bucket'].astype(int)
    actual_agg = actual_df.groupby('bucket', as_index=False).size()
    actual_agg.columns = ['bucket', 'actual']
    actual_sum = actual_agg['actual'].sum()
    actual_agg['actual_distribution'] = actual_agg['actual']/actual_sum

    bound_low = breakpoints[:-1]
    bound_high = breakpoints[1:]
    bounds = pd.DataFrame(list(zip(labels, bound_low, bound_high)), columns=['bucket', 'bound_low', 'bound_high'])
    bounds['bounds'] = '(' + bounds['bound_low'].round(5).astype(str) + ', ' + bounds['bound_high'].round(5).astype(str) + ']'
    bounds = bounds[['bucket', 'bounds']]
    new_row = {'bucket': [0], 'bounds': ['NULL']}
    new_row = pd.DataFrame(new_row)
    bounds = pd.concat([bounds, new_row], axis=0).reset_index(drop=True)

    psi = pd.merge(expected_agg, actual_agg, on='bucket', how='outer')
    psi = psi.fillna({'expected': 0, 'actual':0, 'expected_distribution': 0, 'actual_distribution': 0})
    psi = psi.replace({'expected_distribution': 0, 'actual_distribution': 0}, 0.0001)
    psi = pd.merge(psi, bounds, on='bucket', how='left')
    psi = psi[['bucket', 'bounds', 'expected', 'actual', 'expected_distribution','actual_distribution']]
    psi['psi'] = (psi['actual_distribution'] - psi['expected_distribution']) * np.log(psi['actual_distribution'] / psi['expected_distribution'])
    psi = psi.sort_values('bucket')

    return psi.reset_index(drop=True)

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def calculate_psi(expected: pd.Series,
                  actual: pd.Series,
                  buckets: int = 10
                  ) -> float:

    psi_df = psi_dataframe(expected, actual, buckets)
    return psi_df['psi'].sum()

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def plot_psi(expected: pd.Series,
            actual: pd.Series,
            buckets: int = 10
            ) -> None:

    psi_df = psi_dataframe(expected, actual, buckets)
    psi_df = psi_df[['bucket', 'expected_distribution', 'actual_distribution']]
    psi_df = psi_df.rename(columns={'bucket': 'Bucket', 'expected_distribution': 'Initial population', 'actual_distribution': 'New population'})
    psi_melted = psi_df.melt(id_vars='Bucket', var_name='Population', value_name='Percent')
    p = sns.barplot(x='Bucket', y='Percent', hue='Population', data=psi_melted)
    p.set(xlabel='Bucket', ylabel='Population percent')
    sns.despine(left=True)

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def unique_values(
    df: pd.DataFrame
    , column: str
    ) -> (list, int):

    df_selected = df[[column]]
    df_not_null = df_selected[df_selected[column].notnull()]
    unique_values = df_not_null[column].unique()
    unique_values_list = unique_values.tolist()
    unique_values_list.sort()
    len_values = len(unique_values_list)

    return unique_values_list, len_values

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def create_numeric_classing_rules(
    column: str
    , column_classing: str
    , quantiles: list
    , null_separate: bool = True
    ) -> pd.DataFrame:

    low_range = quantiles[:-1]
    high_range = quantiles[1:]
    last_elem = len(high_range)

    rules_dict = dict()
    all_rules = list()

    if null_separate:
        null_class = 0
        all_rules.append(f'{column} IS NULL')
        rules_dict["Class"] = [null_class]
        rules_dict["Low"] = [None]
        rules_dict["High"] = [None]

        for i, (low, high) in enumerate(zip(low_range, high_range), start = 1):

            if i == 1:
                all_rules.append(f'{column} <= {high}')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)

            if (i > 1) and (i < last_elem):
                all_rules.append(f'({column} > {low} AND {column} <= {high})')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)

            if (i > 1) and (i == last_elem):
                all_rules.append(f'{column} > {low}')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)
    else:
        rules_dict["Class"] = []
        rules_dict["Low"] = []
        rules_dict["High"] = []

        for i, (low, high) in enumerate(zip(low_range, high_range), start = 1):

            if i == 1:
                all_rules.append(f'({column} <= {high} OR {column} IS NULL)')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)

            if (i > 1) and (i < last_elem):
                all_rules.append(f'({column} > {low} AND {column} <= {high})')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)

            if (i > 1) and (i == last_elem):
                all_rules.append(f'{column} > {low}')
                rules_dict["Class"].append(i)
                rules_dict["Low"].append(low)
                rules_dict["High"].append(high)

    rules_dict["Attribute"] = all_rules
    rules_df = pd.DataFrame.from_dict(rules_dict)
    rules_df["Categories"] = None
    rules_df["Name"] = column
    rules_df["ClassingName"] = column_classing
    rules_df["DataType"] = 'numeric'
    rules_df = rules_df[["Class", "Name", "ClassingName", "DataType", "Attribute", "Categories", "Low", "High"]]
    rules_df = rules_df.sort_values(by='Class')

    return rules_df.reset_index(drop=True)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def fine_classing_numeric(
    df: pd.DataFrame
    , column: str
    , fine_classes: int = 10
    , significance: int = 5
    , null_separate: bool = False
    ) -> pd.DataFrame:

    quantiles, quanti_len = unique_values(df, column)

    if (quanti_len <= fine_classes):
        quantiles.append((-float("inf")))
        quantiles.append(float("inf"))
        deduplicated_quantiles = list(set(quantiles))
        deduplicated_quantiles.sort()
    else:
        points = list(np.arange(1, fine_classes, 1)/fine_classes)
        quantiles = df[column].quantile(points, interpolation = 'midpoint')
        quantiles_round = [round(quantil, significance) for quantil in quantiles]
        quantiles_round.append((-float("inf")))
        quantiles_round.append(float("inf"))
        deduplicated_quantiles = list(set(quantiles_round))
        deduplicated_quantiles.sort()

    column_fc = 'fc_' + column

    return create_numeric_classing_rules(column, column_fc, deduplicated_quantiles, null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def create_string_classing_rules(
    column: str
    , column_fc: str
    , df_groups: pd.DataFrame
    , null_separate: bool = True
    ) -> pd.DataFrame:

    if null_separate:
        df_groups["Class"] = df_groups.index + 1
        df_groups["Categories"] = "'" + df_groups[column].apply(str) + "'"
        df_groups["Attribute"] = column + " IN " + "(" + df_groups["Categories"] + ")"
        null_class = 0
        attr = column + ' IS NULL'
        null_row = {'Categories': [None], 'Class': [null_class], 'Attribute': [attr]}
        null_case = pd.DataFrame.from_dict(null_row)
        rules_df = pd.concat([df_groups, null_case]).sort_values(by='Class')
        rules_df['Categories'] = rules_df['Categories'].apply(lambda x: [x])
        rules_df['Name'] = column
        rules_df['ClassingName'] = column_fc
        rules_df['DataType'] = 'string'
        rules_df['Low'] = np.NaN
        rules_df['High'] = np.NaN
        rules_df = rules_df[["Class", "Name", "ClassingName", "DataType", "Attribute", "Categories", "Low", "High"]]
    else:
        df_groups["Class"] = df_groups.index + 1
        df_groups["Categories"] = "'" + df_groups[column].apply(str) + "'"
        df_groups["Attribute"] = np.where(df_groups["Class"] == 1, "(" + column + " IN " + "(" + df_groups["Categories"] + ") OR " + column + " IS NULL)", column + " IN " + "(" + df_groups["Categories"] + ")")
        rules_df = df_groups.sort_values(by='Class')
        rules_df['Categories'] = rules_df['Categories'].apply(lambda x: [x])
        rules_df['Name'] = column
        rules_df['ClassingName'] = column_fc
        rules_df['DataType'] = 'string'
        rules_df['Low'] = np.NaN
        rules_df['High'] = np.NaN
        rules_df = rules_df[["Class", "Name", "ClassingName", "DataType", "Attribute", "Categories", "Low", "High"]]

    rules_df = rules_df.sort_values(by='Class')

    return rules_df.reset_index(drop=True)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def fine_classing_string(
    df: pd.DataFrame
    , column: str
    , target_col: str
    , null_separate: bool = False
    , target_ascending: bool = False
) -> pd.DataFrame:

    df_groups = df.groupby(column, as_index = False, dropna = True)[target_col].mean()
    df_groups = df_groups.sort_values(by = target_col, ascending = target_ascending)
    df_groups = df_groups.reset_index(drop = True)
    column_fc = 'fc_' + column

    return create_string_classing_rules(column, column_fc, df_groups, null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def coarse_classing_numeric(
    df: pd.DataFrame
    , column: str
    , target_col: str
    , coarse_classes: int = 3
    , criterion: str = 'gini'
    , cnst_min_prc: float = 0.05
    , significance: int = 5
    , null_separate: bool = False
    ) -> pd.DataFrame:

    quantiles, quanti_len = unique_values(df, column)

    if (quanti_len <= coarse_classes):
        quantiles.append((-float("inf")))
        quantiles.append(float("inf"))
        deduplicated_quantiles = list(set(quantiles))
        deduplicated_quantiles.sort()
    else:
        df_selected = df[[column, target_col]]
        df_not_null = df_selected[df_selected[column].notnull()]

        X = df_not_null[column].values.reshape(-1, 1)
        y = df_not_null[target_col].values

        n = df_not_null.shape[0]
        min_obs_leaf = math.ceil(n * cnst_min_prc)

        model = DecisionTreeClassifier(criterion = criterion
                                    , max_leaf_nodes = coarse_classes
                                    , min_samples_leaf = min_obs_leaf
                                    , random_state = 42)

        model.fit(X, y)

        tree_rules = tree.export_text(model, feature_names= [column])
        points = []
        for text in tree_rules.splitlines():
            if column in text:
                text = text.strip()
                s_pos = text.rfind(' ')
                text = text[s_pos:]
                f_val = float(text.strip())
                points.append(f_val)

        points_round = [round(point, significance) for point in points]
        points_round.append((-float("inf")))
        points_round.append(float("inf"))
        deduplicated_points = list(set(points_round))
        deduplicated_points.sort()

    column_fc = 'cc_' + column

    return create_numeric_classing_rules(column, column_fc, deduplicated_points, null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def coarse_classing_string(
    df: pd.DataFrame
    , column: str
    , target_col: str
    , coarse_classes: int = 3
    , target_ascending: bool = False
    , null_separate: bool = False
    ) -> pd.DataFrame:

    elements, elements_len = unique_values(df, column)

    if (elements_len <= coarse_classes):
        return fine_classing_string(df, column, target_col, null_separate)
    else:
        woe_table = calculate_woe(df, column, target_col)
        woe_table['WoE'] = woe_table['WoE'].fillna(0)

        X = woe_table['WoE'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters = coarse_classes, random_state = 42, n_init = "auto").fit(X)
        woe_table['Cluster'] = kmeans.predict(X)

        df_groups = woe_table.groupby("Cluster", as_index = False, dropna = False).agg({"Count": "sum", "Events": "sum"})
        df_groups["Bad_Rate"] =  df_groups["Events"] / df_groups["Count"]
        df_groups = df_groups.sort_values(by = "Bad_Rate", ascending = target_ascending)
        df_groups = df_groups.reset_index(drop = True)
        df_groups["Class"] = df_groups.index + 1
        df_groups = df_groups[["Cluster", "Class"]]
        df_merged = pd.merge(woe_table, df_groups, on='Cluster', how='inner')
        df_groups = df_merged.groupby('Class', as_index = False)['Category'].agg(list)
        df_groups["Category_str"] = df_groups['Category'].apply(lambda x: ', '.join([f"'{str(i)}'" for i in x]))
        df_groups[column] = df_groups["Category_str"].str.strip('"\'')
        df_groups = df_groups.drop(['Category', 'Category_str'], axis = 1)
        column_cc = 'cc_' + column

        return create_string_classing_rules(column, column_cc, df_groups, null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.