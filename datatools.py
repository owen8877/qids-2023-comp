import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import metrics
from pandas import Series, DataFrame


def standardize_pd(Xin):
    """
    Parameters
    ----------
    Xin : <pd.DataFrame> dataframe containing the dataset 

    Returns
    -------
    <pd.DataFrame> the column-wise standardized dataset
    """

    return Xin - Xin.min() / (Xin.max() - Xin.min())


def make_clean_data(pd_data, verbose=False):
    """
    -Given any dataframe, removes rows with nan or none, examining 
    feature by feature 
    -This probably can be done by doing the whole dataframe simultaneously 
    """
    # infer the features we have kept by checking columns of the input dataframe

    # Needed: check all features for missing values and remove everybody with missing.
    # check remaining percentage

    features = pd_data.columns
    missing_dict = dict()
    # bad_rows = list()

    bad_rows_index = pd_data.isna().sum(axis=1).to_numpy().nonzero()
    bad_feature_index = pd_data.isna().sum().to_numpy().nonzero()
    bad_feature = pd_data.columns[bad_feature_index]

    for ft in bad_feature:
        # Calculate the percentage of that feature which was True under .isnull()
        num_missing_bool = pd_data.isna().sum()[ft]
        missing_dict[ft] = num_missing_bool / len(pd_data[ft])

        if verbose:
            print("Issue Feature:\n", ft, '\n',
                  '\n Num of null=', num_missing_bool, '\n\n')
        else:
            pass

    # for ft in features:
    #     pd_data.isna().sum()
    #     feature_series = pd_data[ft]
    #     missing_bool = feature_series.isnull()
    #     bad_indices = feature_series.index[missing_bool]
    #     #Calculate the percentage of that feature which was True under .isnull()
    #     missing_dict[ft] = 100*float(np.sum(missing_bool)/feature_series.shape[0])
    #
    #
    #     if not bad_indices.empty:
    #         if verbose:
    #             print("Issue Feature:\n", ft,'\n', bad_indices, '\n Num of null=', len(bad_indices), '\n\n')
    #             bad_rows += list(bad_indices)
    #             print('Here are Nan Indices:', bad_indices)
    #         else:
    #             pass

    # Total percentage(s) of data removed
    if verbose:
        # print('Here are Nan Row Indices:', bad_rows_index[0], '\n') #maybe we don't need but I added here
        print('Total Number of Removed Row Instances = ',
              len(bad_rows_index[0]), '\n ')
        print('Percentage of Removed Features: \n', missing_dict)
    # Eliminate duplicates and sort
    # bad_rows = list(set(bad_rows))
    # bad_rows.sort()

    # Get rid of rows containing null or empty
    # clean_data = pd_data.drop(bad_rows)
    clean_data = pd_data.drop(bad_rows_index[0])

    # Check if clean
    assert np.size(clean_data.isna().sum(axis=1).to_numpy().nonzero()[
                       0]) == 0, "Clean data still contains NaN"

    # Check the number of resulting data points
    # if verbose:
    #     print('Here is shape of original data:',data.shape,'\n\n')
    #     print('Here is shape of the clean data:', data_clean.shape,\
    #           '\n Number of Removed Instances =',len(bad_rows))

    return clean_data, missing_dict, bad_rows_index


def remove_quantiles(pd_data, p=1):
    percentile = p
    quantile = percentile / 100
    remove_indices = list()
    # feature_stats = dict()

    for feature in pd_data.columns:
        feature_series = pd_data[feature]
        quantile_filter = np.quantile(feature_series, [quantile, 1 - quantile])
        feature_outside = feature_series[(feature_series < quantile_filter[0]) | (
                feature_series > quantile_filter[1])]
        # outside_indices = feature_outside.index
        remove_indices += list(feature_outside.index)
    remove_indices = list(set(remove_indices))
    remove_indices.sort()

    pd_data_reduced = pd_data.drop(remove_indices)

    # Calculate what percent of total data is captured in these indices
    percent_removed = 100 * (len(remove_indices) / pd_data.index.shape[0])
    print('Percent of Data Removed Across These Quantiles Is: ', percent_removed)

    return pd_data_reduced, percent_removed


def run_svd(pd_data, percent_var=95):
    # def run_svd(data, rank = 3):
    """
    :pd_data: the dataframe containing the 'already standardized'] data 
    :percent_var: float - a value between [0,100]
    """
    # add checking if percent_var between 0 and 100

    # Calculate the desired number of SVD components in the decomposition
    start_rank = (pd_data.shape[-1] - 1)
    # Make instance of SVD object class from scikit-learn and run the decomposition
    # Issue: scikitlearn TruncatedSVD only allows n_components < n_features (strictly)

    SVD = TruncatedSVD(n_components=start_rank)
    SVD.fit(pd_data)
    X_SVD = SVD.transform(pd_data)

    # Wrap the output as a dataframe
    X_SVD = pd.DataFrame(
        X_SVD, columns=['Singular Component ' + str(i + 1) for i in range(X_SVD.shape[-1])])

    # Calculate the number of components needed to reach variance threshold
    var_per_comp = SVD.explained_variance_ratio_

    # Calculate the total variance explainend in the first k components
    total_var = 100 * np.cumsum(var_per_comp)
    print('------------- SVD Output ----------------')
    print('Percent Variance Explained By First ' +
          str(start_rank) + ' Components: ', total_var, '\n\n')
    # rank = np.nonzero(total_var>=var_threshold)[0][0]+1
    rank = (next(x for x, val in enumerate(total_var) if val > percent_var))
    rank += 1

    if rank == 0:
        print('No quantity of components leq to ' + str(start_rank + 1) +
              ' can explain ' + str(percent_var) + '% variance.')
    else:
        print(str(total_var[rank - 1]) + '% variance ' + 'explained by ' + str(rank) + ' components. ' +
              'Variance Threshold Was ' + str(percent_var) + '.\n\n')

    return X_SVD, rank, percent_var, total_var


def data_quantization(pd_data, scale=10):
    """
    Quantize a panda data frame into integer with new features according to the given scale.
    e.g. if scale = 10: the new feature assign label 1 to the first, and 10 to the last
    :param pd_data:
    :param scale:
    :return: data_quantile: the quantized data
             percent_of_zero: at least that much percent of feature are zeros
    """
    p = np.linspace(0, scale, scale + 1) * 0.1
    data_quantile = pd_data.copy()
    percent_of_zero = {}
    eps = 1e-5

    for feature in pd_data.columns:
        feature_new = feature + '_QUANTILE'
        data_quantile[feature_new] = 0

        for (i, quantile) in enumerate(p[:-1]):
            quantile_filter = np.quantile(
                pd_data[feature], [quantile, p[i + 1]])
            data_quantile.loc[((pd_data[feature] > quantile_filter[0]) &
                               (pd_data[feature] <= quantile_filter[1])), feature_new] = i + 1

            # deal with 0-quantile being non-zero
            if i == 0 and quantile_filter[0] > 0:
                data_quantile.loc[((pd_data[feature] >= quantile_filter[0]) &
                                   (pd_data[feature] <= quantile_filter[1])), feature_new] = i + 1

            if quantile_filter[0] <= eps and quantile_filter[1] >= eps:
                percent_of_zero[feature] = quantile
            elif quantile_filter[0] > eps and i == 0:
                percent_of_zero[feature] = 0

        data_quantile.drop(columns=feature, axis=1, inplace=True)
    return data_quantile, percent_of_zero


def extract_market_data(m_df: DataFrame):
    """
    Input the market data and extract mean price using close
    prices, volatility and mean volume
    :param m_df: [pd.DataFrame] market data, set_index already
    :return: [pd.DataFrame] extracted features from market data
    """
    # sort indexing
    if m_df.index.names == ['asset', 'day', 'timeslot']:
        m_df = m_df.swaplevel(0, 1)
    elif m_df.index.names == ['day', 'asset', 'timeslot']:
        pass
    else:
        raise ValueError('Unsupported index ordering')

    m_df.sort_index(ascending=True, inplace=True)

    m_df_day = m_df.groupby(level=[0, 1])[['volume', 'money']].sum()
    # Compute average price
    m_df_day['avg_price'] = m_df_day['money'] / m_df_day['volume']
    # find index of zero volume
    indx_day = m_df_day[m_df_day['volume'] == 0].index
    # compute replacing value as mean of high and low
    for (i, indx) in enumerate(indx_day):
        replace_value = .5 * m_df.loc[indx, 'high'].max() + .5 * m_df.loc[indx, 'low'].min()
        m_df_day.loc[indx, 'avg_price'] = replace_value

    assert np.size(m_df_day.isna().sum(axis=1).to_numpy().nonzero()[
                       0]) == 0, "Clean data still contains NaN"

    # Compute volatility
    T = 50.0  # number of time units
    # note numpy use 0 dof while pd use 1 dof
    m_df_day['volatility'] = m_df.groupby(level=[0, 1])['close'].std() * np.sqrt(T)

    # Compute average volume:
    m_df_day['mean_volume'] = m_df_day['volume'] / T

    # drop unnecessary features:
    m_df_day = m_df_day.drop(columns=['volume', 'money'])

    return m_df_day


if __name__ == "__main__":
    pass
