
def outlier_removal(data_dict):
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(data_dict)