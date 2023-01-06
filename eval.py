import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from feature_engine.encoding import WoEEncoder
from sklearn.impute import KNNImputer
import pandas as pd
import pickle

ref_dict = {
    "A": ["measurement_5", "measurement_6", "measurement_8"],
    "B": ["measurement_4", "measurement_5", "measurement_7"],
    "C": ["measurement_5", "measurement_7", "measurement_8", "measurement_9"],
    "D": ["measurement_5", "measurement_6", "measurement_7", "measurement_8"],
    "E": ["measurement_4", "measurement_5", "measurement_6", "measurement_8"],
    "F": ["measurement_4", "measurement_5", "measurement_6", "measurement_7"],
    "G": ["measurement_4", "measurement_6", "measurement_8", "measurement_9"],
    "H": [
        "measurement_4",
        "measurement_5",
        "measurement_7",
        "measurement_8",
        "measurement_9",
    ],
    "I": ["measurement_3", "measurement_7", "measurement_8"],
}


def preprocess(train, test, features):
    data = pd.concat([train, test])
    # new features
    data["m3_missing"] = data["measurement_3"].isnull().astype(np.int8)
    data["m5_missing"] = data["measurement_5"].isnull().astype(np.int8)
    data["area"] = data["attribute_2"] * data["attribute_3"]
    label = [f"measurement_{i:d}" for i in range(3, 17)]
    data["avg_m3_m16"] = np.mean(data[label], axis=1)
    data["std_m3_m16"] = np.std(data[label], axis=1)
    data["loading"] = np.log(data["loading"])
    data["measurement_2"].clip(11, None)

    for code in data.product_code.unique():
        cur_data = data[data.product_code == code]
        cur_ref = ref_dict[code]

        train_x = cur_data[cur_ref + ["measurement_17"]].dropna(how="any")

        test_x = cur_data[
            (cur_data[cur_ref].isnull().sum(axis=1) == 0)
            & (cur_data["measurement_17"].isnull())
        ]

        model = HuberRegressor(epsilon=1.9)
        model.fit(train_x[cur_ref], train_x["measurement_17"])

        data.loc[
            (data.product_code == code)
            & (data[cur_ref].isnull().sum(axis=1) == 0)
            & (data["measurement_17"].isnull()),
            "measurement_17",
        ] = model.predict(test_x[cur_ref])

        knn = KNNImputer(n_neighbors=3)
        data.loc[data.product_code == code, features] = knn.fit_transform(
            data.loc[data.product_code == code, features]
        )

    train = data[data["failure"].notnull()]
    test = data[data["failure"].isnull()].drop(["failure"], axis=1)

    x = train.drop(["failure"], axis=1)
    y = train["failure"].astype(int)

    # use the woe_encoder trained by train_data
    woe_encoder = WoEEncoder(variables=["attribute_0"])
    woe_encoder.fit(x, y)
    x = woe_encoder.transform(x)
    test = woe_encoder.transform(test)

    return x, test


def scaling(train, test, features):
    scaler = StandardScaler()
    scaler.fit(train[features])
    scaled_test = scaler.transform(test[features])

    # transfer to dataframe
    new_test = test.copy()
    new_test[features] = scaled_test

    assert len(test) == len(new_test)
    return new_test


def regression(model, train, test, features):
    final_result = np.zeros(len(test))

    x_test = scaling(train, test, features)
    final_result = model.predict_proba(x_test[features])[:, 1]

    return final_result


if __name__ == "__main__":
    train = pd.read_csv("train/train.csv")
    test = pd.read_csv("train/test.csv")
    features_preprocess = [
        feature
        for feature in test.columns
        if feature.startswith("measurement") or feature == "loading"
    ]

    # Comment this if you don't need the preprocessing
    # Make sure that there is no invalid data
    train, test = preprocess(train, test, features_preprocess)
    result = np.zeros(len(test))

    feature_used = [
        "loading",
        "attribute_0",
        "area",
        "measurement_17",
        "m3_missing",
        "m5_missing",
        "measurement_0",
        "measurement_1",
        "measurement_2",
    ]

    models = []
    # load models
    for i in range(5):
        with open(f"model/model_{i+1}.pkl", "rb") as f:
            models.append(pickle.load(f))
    # predict the results
    for model in models:
        result += regression(model, train, test, feature_used) / 5

    # write to csv
    submission = pd.read_csv("train/sample_submission.csv")
    submission["failure"] = result
    submission.to_csv("submission.csv", index=False)
