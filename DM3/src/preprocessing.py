import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

class PreprocessDataTrainTestSplit:
    """Class to process samples and split them in train-test
    """
    def __init__(self,
                 data_path,
                 split_date,
                 columns_to_del=['IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS'],
                 normalize=False, 
                 max_date=None):
        """Class constructor

        Args:
            data_path (str): data path
            split_date (str): last date of training sample
            columns_to_del (list, optional): columns to delete. Defaults to ['IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS'].
            normalize (bool, optional): if True, samples are normalized. Defaults to False.
            max_date (_type_, optional): last date of dataset. Defaults to None.
        """
        
        self.data_path = data_path
        self.split_date = split_date
        self.columns_to_del = columns_to_del
        self.normalize = normalize
        self.max_date = max_date
    
    def process(self):
        """Apply preprocessing and train-test division

        Returns:
            Tuple[pd.DataFrame, pd.Dataframe]: preprocessed train and test datasets
        """
        df = pd.read_csv(self.data_path)
        df = df.iloc[1:,:]
        df["sasdate"] = pd.to_datetime(df["sasdate"], format='%m/%d/%Y')
        df.set_index("sasdate", inplace=True)
        if (self.columns_to_del is not None)&(self.max_date is not None):
            df_filtered = df.drop(columns=self.columns_to_del)
            df_filtered = df_filtered.loc[df.index <= self.max_date]
        elif (self.columns_to_del is not None)&(self.max_date is None):
            df_filtered = df.drop(columns=self.columns_to_del)
        elif (self.columns_to_del is None)&(self.max_date is not None):
            df_filtered = df_filtered.loc[df.index <= self.max_date]
        else:
            df_filtered = df.copy()

        train_df = df_filtered[df_filtered.index <= self.split_date]
        test_df = df_filtered[df_filtered.index > self.split_date]

        df_y_train = train_df["INDPRO"]
        df_x_train = train_df.drop('INDPRO', axis=1).copy()

        df_y_test = test_df["INDPRO"]
        df_x_test = test_df.drop('INDPRO', axis=1).copy()

        ##Step1 : Process nan values
        #Drop columns with too many nan
        col_to_drop = ['ACOGNO', 'TWEXMMTH', 'UMCSENTx', 'ANDENOx']
        column_transformer = ColumnTransformer(
            transformers=[('drop_columns', 'drop', col_to_drop)],
            remainder='passthrough'
        )
        #Replace nan values by median for others
        imputer = SimpleImputer(strategy='median')

        #Convert numpy arrays to Pandas DataFrame
        to_df = FunctionTransformer(lambda x: pd.DataFrame(x, columns=df_x_train.columns.drop(col_to_drop)), validate=False)

        pipeline_drop_imputer = Pipeline([
            ('preprocessor', column_transformer),
            ('imputer', imputer),
            ('to_dataframe', to_df)
        ])

        df_x_train_transform = pipeline_drop_imputer.fit_transform(df_x_train)
        df_x_test_transform = pipeline_drop_imputer.transform(df_x_test)

        df_x_train_transform["sasdate"] = df_x_train.index
        df_x_train_transform.set_index("sasdate", inplace=True)
        df_x_test_transform["sasdate"] = df_x_test.index
        df_x_test_transform.set_index("sasdate", inplace=True)

        df_train_transform = df_x_train_transform.copy()
        df_train_transform["INDPRO"] = df_y_train.values
        df_test_transform = df_x_test_transform.copy()
        df_test_transform["INDPRO"] = df_y_test.values

        ##Step 2 : first-order differenciation
        df_train_transform_diff = df_train_transform.diff().drop(index=df_train_transform.index[0], axis=0, inplace=False)
        df_test_transform_diff = df_test_transform.diff().drop(index=df_test_transform.index[0], axis=0, inplace=False)

        if self.normalize:
        ##Step3: features standardization
            df_y_train_transform_diff = df_train_transform_diff["INDPRO"]
            df_x_train_transform_diff = df_train_transform_diff.drop('INDPRO', axis=1).copy()

            df_y_test_transform_diff = df_test_transform_diff["INDPRO"]
            df_x_test_transform_diff = df_test_transform_diff.drop('INDPRO', axis=1).copy()

            scaler = StandardScaler()
            to_df = FunctionTransformer(lambda x: pd.DataFrame(x, columns=df_x_train.columns.drop(col_to_drop)), validate=False)

            pipeline_scaler = Pipeline([
                ('scaler', scaler),
                ('to_dataframe', to_df)
            ])

            df_x_train_clean_diff_stand = pipeline_scaler.fit_transform(df_x_train_transform_diff)
            df_x_test_clean_diff_stand = pipeline_scaler.transform(df_x_test_transform_diff)

            df_x_train_clean_diff_stand["sasdate"] = df_x_train_transform_diff.index
            df_x_train_clean_diff_stand.set_index("sasdate", inplace=True)
            df_x_test_clean_diff_stand["sasdate"] = df_x_test_transform_diff.index
            df_x_test_clean_diff_stand.set_index("sasdate", inplace=True)

            df_train_clean_diff_stand = df_x_train_clean_diff_stand.copy()
            df_train_clean_diff_stand["INDPRO"] = df_y_train_transform_diff.values
            df_test_clean_diff_stand = df_x_test_clean_diff_stand.copy()
            df_test_clean_diff_stand["INDPRO"] = df_y_test_transform_diff.values

            return df_train_clean_diff_stand, df_test_clean_diff_stand
        
        else:
            return df_train_transform_diff, df_test_transform_diff