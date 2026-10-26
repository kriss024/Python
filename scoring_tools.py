'''
Created on 11-10-2026

@author   : Krzysztof Bruszewski
@email    : krzysztof.bruszewski@gmail.com
@file     : scoring_tools.py
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
    category_counts['Name'] = cat_variable
    category_counts['Count'] = category_counts['Events'] + category_counts['Non-Events']
    category_counts['Bad_Rate'] = category_counts['Events'] / category_counts['Count']
    category_counts = category_counts[['Name', 'Category', 'Count', 'Events', 'Bad_Rate', 'Non-Events']]
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

def rules_table_to_sql(df_classing_woe: pd.DataFrame,
                       column: str = 'Class',
                       else_val: Any = -1,
                       new_line_char: str = '\t'
                       ) -> str:
    sql_list = []
    sql_list.append('CASE')

    for index, row in df_classing_woe.iterrows():
        outcome = row[column]
        if not isinstance(outcome, str):
            outcome = str(outcome)

        sql_case = f"WHEN {row['Attribute']} THEN {outcome}"
        sql_list.append(sql_case)

    if else_val is not None:
        sql_list.append(f'ELSE {else_val}')

    classing_name = df_classing_woe['ClassingName'].iloc[0]
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

def pandas_unique_values(
    df: pd.DataFrame
    , column: str
    ) -> list:

    return df[column]\
        .dropna().unique().tolist()

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

def calculate_quantiles(
    df: pd.DataFrame
    , column: str
    , buckets: int = 20
    , significance: int = 5
    ) -> list:

    points = list(np.arange(1, buckets, 1)/buckets)
    quantiles = df[column].dropna().quantile(points, interpolation = 'midpoint')
    quantiles = np.array(quantiles)
    quantiles = np.round(quantiles, decimals=significance)
    quantiles = np.append(quantiles, -float("inf"))
    quantiles = np.append(quantiles, float("inf"))
    quantiles = np.unique(quantiles)
    quantiles = np.sort(quantiles)

    return quantiles

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def fine_classing_numeric(
    df: pd.DataFrame
    , column: str
    , fine_classes: int = 10
    , significance: int = 5
    , null_separate: bool = False
    ) -> pd.DataFrame:

    unique_values = pandas_unique_values(df, column)

    if (len(unique_values) <= fine_classes):
        breakpoints = np.array(unique_values)
        breakpoints = np.append(breakpoints, -float("inf"))
        breakpoints = np.append(breakpoints, float("inf"))
        breakpoints = np.unique(breakpoints)
        breakpoints = np.sort(breakpoints)

        return create_numeric_classing_rules(column, f"fc_{column}", breakpoints.tolist(), null_separate)

    else:
        quantiles = calculate_quantiles(df, column, fine_classes, significance)

        return create_numeric_classing_rules(column, f"fc_{column}", quantiles.tolist(), null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def create_string_classing_rules(
    column: str
    , column_classing: str
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
        rules_df['ClassingName'] = column_classing
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
        rules_df['ClassingName'] = column_classing
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

    return create_string_classing_rules(column, f"fc_{column}", df_groups, null_separate)

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

    unique_values = pandas_unique_values(df, column)

    if (len(unique_values) <= coarse_classes):
        breakpoints = np.array(unique_values)
        breakpoints = np.append(breakpoints, -float("inf"))
        breakpoints = np.append(breakpoints, float("inf"))
        breakpoints = np.unique(breakpoints)
        breakpoints = np.sort(breakpoints)

        return create_numeric_classing_rules(column, f"cc_{column}", breakpoints.tolist(), null_separate)
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
        breakpoints = np.array([])
        for text in tree_rules.splitlines():
            if column in text:
                text = text.strip()
                s_pos = text.rfind(' ')
                text = text[s_pos:]
                f_val = float(text.strip())
                breakpoints = np.append(breakpoints, f_val)

        breakpoints = np.round(breakpoints, decimals=significance)
        breakpoints = np.append(breakpoints, -float("inf"))
        breakpoints = np.append(breakpoints, float("inf"))
        breakpoints = np.unique(breakpoints)
        breakpoints = np.sort(breakpoints)

        return create_numeric_classing_rules(column, f"cc_{column}", breakpoints.tolist(), null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def coarse_classing_string(
    df: pd.DataFrame
    , column: str
    , target_col: str
    , coarse_classes: int = 3
    , target_ascending: bool = False
    , null_separate: bool = False
    ) -> pd.DataFrame:

    unique_values = pandas_unique_values(df, column)

    if (len(unique_values) <= coarse_classes):
        df_classing = fine_classing_string(df, column, target_col, null_separate)
        df_classing["ClassingName"] = f"cc_{column}"

        return df_classing
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

        return create_string_classing_rules(column, f"cc_{column}", df_groups, null_separate)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

import statsmodels.api as sm

def logistic_regression(df: pd.DataFrame,
                        indep_vars: list,
                        target_name: str
                        ) -> sm.Logit:

    x = df.loc[:, indep_vars]
    X = sm.add_constant(x, has_constant = "add")
    y = df[target_name].values
    model = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100)

    return model

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

import logging
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats.contingency import association
from sklearn.metrics import roc_auc_score

def logistic_regression_diagnostics(included_features: list,
                                    df_train: pd.DataFrame,
                                    df_test: pd.DataFrame,
                                    target: str
                                    ) -> (dict, sm.Logit):
    """
    Fit a logistic regression model for a selected feature set and compute
    performance, coefficient, and multicollinearity diagnostics.
    """
    try:
        feats = list(included_features)
        X_train = df_train[feats]
        y_train = df_train[target].values
        X_test  = df_test[feats]
        n_cols  = len(feats)

        X_train_const = sm.add_constant(X_train, has_constant="add")
        model = sm.Logit(y_train, X_train_const).fit(disp=0, method="newton", maxiter=100)

        # Coefficients (including intercept)
        params = model.params
        pvalues = model.pvalues
        max_coef = params.max()
        max_p_value = pvalues.max()
        all_neg_betas = int(max_coef < 0)

        # Predictions
        proba_train = model.predict(X_train_const)
        proba_test  = model.predict(sm.add_constant(X_test, has_constant="add"))

        # Discriminatory power
        auc_train = roc_auc_score(y_train, proba_train)
        auc_test  = roc_auc_score(df_test[target], proba_test)
        gini_train = 2 * auc_train - 1
        gini_test  = 2 * auc_test - 1
        delta_gini = abs(gini_train - gini_test)

        if len(included_features) < 2:
            return {
                "included_features": [", ".join(included_features)],
                "n_included_features": n_cols,
                "all_negative_betas": all_neg_betas,
                "max_pvalue": float(max_p_value),
                "gini_train": float(gini_train),
                "gini_test": float(gini_test),
                "delta_gini": float(delta_gini),
                "max_vif": np.nan,
                "max_con_index": np.nan,
                "max_pearson": np.nan,
                "max_cramer_v": np.nan,
            }, model

        # Multicollinearity
        X = X_train.values.astype(float)
        vif_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_vif = max(vif_list) if vif_list else np.nan

        # Pearson correlation (upper triangle)
        corr = np.corrcoef(X, rowvar=False)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        max_pearson = np.abs(corr[mask]).max() if mask.any() else 0.0

        # Condition number (on column-normalised matrix)
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1.0          # safety
        X_norm = X / norms
        max_con_index = np.linalg.cond(X_norm)

        # Cramér's V correlation (upper triangle)
        corr = np.zeros((n_cols, n_cols))
        np.fill_diagonal(corr, 1.0)

        for i, col1 in enumerate(included_features):
            for j, col2 in enumerate(included_features):
                if i < j:
                    contingency_table = pd.crosstab(X_train[col1], X_train[col2])
                    cramer_v = association(contingency_table.values, method="cramer")
                    corr[i, j] = cramer_v
                    corr[j, i] = cramer_v

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        max_cramer_v = np.abs(corr[mask]).max() if mask.any() else 0.0

        return {
            "included_features": [", ".join(included_features)],
            "n_included_features": n_cols,
            "all_negative_betas": all_neg_betas,
            "max_pvalue": float(max_p_value),
            "gini_train": float(gini_train),
            "gini_test": float(gini_test),
            "delta_gini": float(delta_gini),
            "max_vif": float(max_vif),
            "max_con_index": float(max_con_index),
            "max_pearson": float(max_pearson),
            "max_cramer_v": float(max_cramer_v),
        }, model

    except Exception as e:
        logging.warning("Model %s failed: %s", included_features, e)
        return None, None

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

import seaborn as sns
import matplotlib.pyplot as plt

def classing_plot(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column: str,
    target: str,
    max_bins: int = 10
) -> None:

    train_cp = df_train[[column, target]].copy()
    test_cp = df_test[[column, target]].copy()

    column_binned = f"{column}_binned"
    missing_categories = "Missing"

    is_numeric = None

    numeric_cols = train_cp[[column]].select_dtypes(include="number").columns

    if len(numeric_cols) > 0 and column.lower() == numeric_cols[0].lower():

        unique_list = pandas_unique_values(train_cp, column)

        if len(unique_list) <= max_bins:
            bins = np.array(unique_list)
            bins = np.append(bins, -float("inf"))
            bins = np.append(bins, float("inf"))
            bins = np.unique(bins)
            bins = np.sort(bins)
        else:
            bins = calculate_quantiles(train_cp, column, max_bins)

        train_cp[column_binned] = pd.cut(train_cp[column], bins=bins)

        test_cp[column_binned] = pd.cut(test_cp[column], bins=bins)

        is_numeric = True
    else:
        train_cp[column_binned] = train_cp[column].astype("category")

        test_cp[column_binned] = test_cp[column].astype("category")

        is_numeric = False

    train_cp[column_binned] = train_cp[column_binned].cat.add_categories(
        [missing_categories]
    )

    train_cp[column_binned] = train_cp[column_binned].fillna(missing_categories)

    test_cp[column_binned] = test_cp[column_binned].cat.add_categories(
        [missing_categories]
    )
    test_cp[column_binned] = test_cp[column_binned].fillna(missing_categories)

    train_agg = train_cp.groupby(column_binned, as_index=False, observed=False).agg(
        {target: ["count", "sum", "mean"]}
    )

    train_agg.columns = [column_binned, "count", "sum", "mean"]

    test_agg = test_cp.groupby(column_binned, as_index=False, observed=False).agg(
        {target: ["count", "sum", "mean"]}
    )

    test_agg.columns = [column_binned, "count", "sum", "mean"]

    if is_numeric == False:

        train_bin_mean = train_agg[[column_binned, 'mean']].fillna({'mean': 0.0}).copy()
        test_bin_mean = test_agg[[column_binned, 'mean']].fillna({'mean': 0.0}).copy()

        bin_order = (
            pd.concat([train_bin_mean, test_bin_mean], ignore_index=True)
            .drop_duplicates(subset=[column_binned], keep='first')
            .sort_values('mean')
            .drop(columns='mean')
            .reset_index(drop=True)
            .reset_index(names='col_order')
        )

        train_agg_aligned = (
            bin_order
            .merge(train_agg, on=column_binned, how='left')
            .fillna({'count': 0, 'sum': 0, 'mean': 0.0})
            .sort_values('col_order')
            .drop(columns='col_order')
        )

        test_agg_aligned = (
            bin_order
            .merge(test_agg, on=column_binned, how='left')
            .fillna({'count': 0, 'sum': 0, 'mean': 0.0})
            .sort_values('col_order')
            .drop(columns='col_order')
        )

        train_agg = train_agg_aligned.copy()
        test_agg = test_agg_aligned.copy()

    if missing_categories in train_agg[column_binned].cat.categories:

        train_agg_no_missing = train_agg[train_agg[column_binned] != "Missing"].copy()
        test_agg_no_missing = test_agg[test_agg[column_binned] != "Missing"].copy()

        first_row = train_agg_no_missing.head(1).copy()
        first_row["row"] = "first_row"

        last_row = train_agg_no_missing.tail(1).copy()
        last_row["row"] = "last_row"

        df_first_last_union = pd.concat([first_row, last_row], ignore_index=True)

        df_train_agg_missing = train_agg[train_agg[column_binned] == "Missing"].rename(
            columns={"mean": "mean_missing"}
        ).fillna({"mean_missing": 0.0})

        df_cross_join = pd.merge(
            df_first_last_union, df_train_agg_missing["mean_missing"], how="cross"
        )

        df_cross_join["diff"] = np.abs(df_cross_join["mean"] - df_cross_join["mean_missing"])
        df_cross_join = df_cross_join.sort_values("diff", ascending=True).head(1)

        train_agg_missing = train_agg[train_agg[column_binned] == "Missing"].copy()
        test_agg_missing = test_agg[test_agg[column_binned] == "Missing"].copy()

        belong = df_cross_join["row"][0]

        if belong == "first_row":
            train_agg = pd.concat(
                [train_agg_missing, train_agg_no_missing], ignore_index=True
            )
            test_agg = pd.concat(
                [test_agg_missing, test_agg_no_missing], ignore_index=True
            )
        elif belong == "last_row":
            train_agg = pd.concat(
                [train_agg_no_missing, train_agg_missing], ignore_index=True
            )
            test_agg = pd.concat(
                [test_agg_no_missing, test_agg_missing], ignore_index=True
            )
        else:
            pass


    train_agg = train_agg.reset_index(drop=True).reset_index()
    test_agg = test_agg.reset_index(drop=True).reset_index()

    # Color definition
    PALETTE = {
        "bar": [129 / 255, 179 / 255, 92 / 255],
        "line_train": [0 / 255, 145 / 255, 90 / 255],
        "line_test": [0 / 255, 109 / 255, 67 / 255],
        "ref_line": [129 / 255, 179 / 255, 92 / 255],
        "count_bar": [120 / 255, 135 / 255, 138 / 255],
    }

    fig, ax1 = plt.subplots()

    # 1. Add label to the bar plot
    sns.barplot(
        x="index",
        y="count",
        data=train_agg,
        alpha=0.6,
        ax=ax1,
        color=PALETTE["count_bar"],
        label="Count",
    )
    ax1.bar_label(ax1.containers[0])
    ax1.set_xticks(range(len(train_agg[column_binned])))
    ax1.set_xticklabels(train_agg[column_binned], rotation=45)
    ax1.set(xlabel=None)
    ax1.set_yscale("log")
    ax1.set_ylabel("Count (Log Scale)")
    ax1.set_title(column)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    # 2. Add labels to the line plots
    sns.lineplot(
        x="index",
        y="mean",
        data=train_agg,
        alpha=0.8,
        ax=ax2,
        linewidth=3,
        color=PALETTE["line_train"],
        label="Training set Default Rate",
    )
    sns.lineplot(
        x="index",
        y="mean",
        data=test_agg,
        alpha=0.8,
        ax=ax2,
        linewidth=3,
        color=PALETTE["line_test"],
        label="Test set Default Rate",
    )

    dr_ref_line = train_cp[target].mean()

    ax2.axhline(
        dr_ref_line,
        linewidth=3,
        alpha=0.7,
        color=PALETTE["ref_line"],
        label="Avg. Default Rate (Train)",
    )
    ax2.text(
        0,
        dr_ref_line * 1.05,
        f"Avg. Default Rate (Train): {dr_ref_line * 100 :.2f}%",
        color=PALETTE["ref_line"],
    )
    ax2.set(ylabel="Default Rate")

    # 3. Combine legends from both axes
    plt.legend(loc="upper right")
    plt.show()