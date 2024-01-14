import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

class PreprocessDataTrainTestSplit:
    def __init__(self, data_path, test_size, normalize=False, max_date=None):
        self.data_path = data_path
        self.test_size = test_size
        self.normalize = normalize
        self.max_date = max_date
    
    def process(self):
        df = pd.read_csv(self.data_path)
        df = df.iloc[1:,:]
        df["sasdate"] = pd.to_datetime(df["sasdate"], format='%m/%d/%Y')
        df.set_index("sasdate", inplace=True)
        if self.max_date is not None:
            df_filtered = df.loc[df.index <= self.max_date]
        else:
            df_filtered = df.copy()

        df_y = df_filtered["INDPRO"]
        df_x = df_filtered.drop('INDPRO', axis=1).copy()
        df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=self.test_size, shuffle=False)

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
        to_df = FunctionTransformer(lambda x: pd.DataFrame(x, columns=df_x.columns.drop(col_to_drop)), validate=False)

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
            to_df = FunctionTransformer(lambda x: pd.DataFrame(x, columns=df_x.columns.drop(col_to_drop)), validate=False)

            pipeline_scaler = Pipeline([
                ('scaler', scaler),
                ('to_dataframe', to_df)
            ])

            df_x_train_clean_diff_stand = pipeline_scaler.fit_transform(df_x_train_transform_diff)
            df_x_test_clean_diff_stand = pipeline_scaler.transform(df_x_test_transform_diff)

            df_train_clean_diff_stand = df_x_train_clean_diff_stand.copy()
            df_train_clean_diff_stand["INDPRO"] = df_y_train_transform_diff.values
            df_test_clean_diff_stand = df_x_test_clean_diff_stand.copy()
            df_test_clean_diff_stand["INDPRO"] = df_y_test_transform_diff.values

            return df_train_clean_diff_stand, df_test_clean_diff_stand
        
        else:
            return df_train_transform_diff, df_test_transform_diff

class PreprocessDataTSCV:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def process(self):
        df = pd.read_csv(self.data_path)
        df.drop('sasdate', axis=1, inplace=True)
        df = df.iloc[1:,:]
        df_y = df["INDPRO"]
        df_x = df.drop('INDPRO', axis=1).copy()

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
        to_df = FunctionTransformer(lambda x: pd.DataFrame(x, columns=df_x.columns.drop(col_to_drop)), validate=False)

        pipeline_drop_imputer = Pipeline([
            ('preprocessor', column_transformer),
            ('imputer', imputer),
            ('to_dataframe', to_df)
        ])

        df_x_transform = pipeline_drop_imputer.fit_transform(df_x)
        df_transform = df_x_transform.copy()
        df_transform["INDPRO"] = df_y.values

        ##Step 2 : first-order differenciation

        df_transform_diff = df_transform.diff().drop(index=df_transform.index[0], axis=0, inplace=False)

        return df_transform_diff