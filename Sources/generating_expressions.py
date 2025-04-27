import sympy as smp
import pickle
import pandas as pd
import numpy as np
from pysr import PySRRegressor
import os

SR_model = PySRRegressor(
            niterations=60,  # < Increase me for better results
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[
                # "cos",
                "exp",
                # "sin",
                # "inv(x) = 1/x",
                "exp_neg(x) = exp(-x)",
                # "abs",
                # "sqrt"
            ],
            extra_sympy_mappings={
                # "inv": lambda x: 1 / x,
                "exp_neg": lambda x: smp.exp(-x),
            },
            
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            # ^ Custom loss function (julia syntax)
            warm_start=False,
            verbosity=False,
            progress=False,
            temp_equation_file=True,
            delete_tempfiles=True
        )

countries = [country[0:-11] for country in os.listdir("../RealData/data")]
countries

all_results = pd.DataFrame()

for country in countries:

    data = pd.read_csv(f"../RealData/data/{country}-Mx_1x1.txt", delim_whitespace=True)
    data["Age"] = data["Age"].map(lambda x: 110 if x == "110+" else x)
    data.Age = data.Age.astype("int")

    data.Male = pd.to_numeric(data.Male, errors="coerce")
    data.Female = pd.to_numeric(data.Female, errors="coerce")
    data.Total = pd.to_numeric(data.Total, errors="coerce")

    data = data.dropna()
    

    initial_year, end_year = data.Year.unique().min(), data.Year.unique().max()

    for year in range(initial_year, end_year+1):
        current_results = {}
        
        # Trimmed Data
        data_ = data[(data["Year"] == year) & (data["Age"] >= 30)]
        
        # Log data
        log_total = np.log(data_.Total)
        log_female = np.log(data_.Female)
        log_male = np.log(data_.Male)

        # Create a mask to filter out NaN, inf, and excessively large values
        mask = (~np.isnan(log_total)) & (~np.isinf(log_total)) & (log_total < np.finfo(np.float64).max)
        log_total = log_total[mask]
        age_total = np.array(data_.Age)[mask]

        mask = (~np.isnan(log_female)) & (~np.isinf(log_female)) & (log_female < np.finfo(np.float64).max)
        log_female = log_female[mask]
        age_female = np.array(data_.Age)[mask]

        mask = (~np.isnan(log_male)) & (~np.isinf(log_male)) & (log_male < np.finfo(np.float64).max)
        log_male = log_male[mask]
        age_male = np.array(data_.Age)[mask]

        # plt.figure(figsize=(10, 6))

        # plt.yscale("log")
        # plt.title(year)

        # sns.lineplot(data_, x="Age", y="Total", label="Total")
        # sns.lineplot(data_, x="Age", y="Female", label="Female")
        # sns.lineplot(data_, x="Age", y="Male", label="Male")
        # plt.show()

        for iteration in range(3):
            print(f"Year: {year}  -  Iteration: {iteration}")
            current_results["iteration"] = iteration
            current_results["Year"] = year
            SR_model.fit(np.c_[age_total], np.c_[log_total])
            current_results["Total"] = (SR_model.sympy())
            SR_model.fit(np.c_[age_female], np.c_[log_female])
            current_results["Female"] = (SR_model.sympy())
            SR_model.fit(np.c_[age_male], np.c_[log_male])
            current_results["Male"] = (SR_model.sympy())

            current_df = pd.DataFrame([current_results])
            all_results = pd.concat([all_results, current_df])

    all_results["Country"] = country

    with open(f"../RegressionResults/all_results_log_expand-{country}.pkl", "wb") as file:
        pickle.dump(all_results, file)


with open(f"../RegressionResults/all_results_log_expand-all.pkl", "wb") as file:
    pickle.dump(all_results, file)