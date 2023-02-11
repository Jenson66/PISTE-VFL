
import csv
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
import copy


# split data
def split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len * 0.8)
    train_data = data[:split1]
    test_data = data[split1:]
    return train_data,  test_data


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))



def records_path(acti, dataset, number_client, num_cutlayer):
    csvFile_1 = open(f'Results/{dataset}/num_client{number_client}/VFL_client1_{acti}_c{num_cutlayer}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/{dataset}/num_client{number_client}/VFL_client2_{acti}_c{num_cutlayer}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    return writer_1, writer_2



def load_data(dataset, batch_size):
    if dataset == 'bank_marketing':
        processed_data = 'Data/processed_data/bank-additional-full.csv'
        data = pd.read_csv(processed_data)
        data['age']=(data['age']-data['age'].min())/(data['age'].max()-data['age'].min())
        data['duration']=(data['duration']-data['duration'].min())/(data['duration'].max()-data['duration'].min())
        data['campaign']=(data['campaign']-data['campaign'].min())/(data['campaign'].max()-data['campaign'].min())
        data['pdays']=(data['pdays']-data['pdays'].min())/(data['pdays'].max()-data['pdays'].min())
        data['previous']=(data['previous']-data['previous'].min())/(data['previous'].max()-data['previous'].min())
        data['emp.var.rate']=(data['emp.var.rate']-data['emp.var.rate'].min())/(data['emp.var.rate'].max()-data['emp.var.rate'].min())
        data['cons.price.idx']=(data['cons.price.idx']-data['cons.price.idx'].min())/(data['cons.price.idx'].max()-data['cons.price.idx'].min())
        data['cons.conf.idx']=(data['cons.conf.idx']-data['cons.conf.idx'].min())/(data['cons.conf.idx'].max()-data['cons.conf.idx'].min())
        data['euribor3m']=(data['euribor3m']-data['euribor3m'].min())/(data['euribor3m'].max()-data['euribor3m'].min())
        data['nr.employed']=(data['nr.employed']-data['nr.employed'].min())/(data['nr.employed'].max()-data['nr.employed'].min())
        for i in range(len(data)):
          if (data['poutcome_nonexistent'][i]):
            data['poutcome_failure'][i]=2
          if(data['poutcome_success'][i]):
            data['poutcome_failure'][i]=3
          if (data['job_blue-collar'][i]):
            data['job_admin.'][i]=2
          if(data['job_entrepreneur'][i]):
            data['job_admin.'][i]=3
          if (data['job_housemaid'][i]):
            data['job_admin.'][i]=4
          if(data['job_management'][i]):
            data['job_admin.'][i]=5
          if(data['job_retired'][i]):
            data['job_admin.'][i]=6
          if(data['job_self-employed'][i]):
            data['job_admin.'][i]=7
          if(data['job_services'][i]):
            data['job_admin.'][i]=8
          if(data['job_student'][i]):
            data['job_admin.'][i]=9
          if(data['job_technician'][i]):
            data['job_admin.'][i]=10
          if (data['job_unemployed'][i]):
            data['job_admin.'][i]=11
          if(data['marital_married'][i]):
            data['marital_divorced'][i]=2
          if(data['marital_single'][i]):
            data['marital_divorced'][i]=3
          if(data['contact_telephone'][i]):
            data['contact_cellular'][i]=2
          if (data['month_aug'][i]):
            data['month_apr'][i]=2
          if(data['month_dec'][i]):
            data['month_apr'][i]=3
          if (data['month_jul'][i]):
            data['month_apr'][i]=4
          if(data['month_jun'][i]):
            data['month_apr'][i]=5
          if(data['month_mar'][i]):
            data['month_apr'][i]=6
          if(data['month_may'][i]):
            data['month_apr'][i]=7
          if(data['month_nov'][i]):
            data['month_apr'][i]=8
          if(data['month_oct'][i]):
            data['month_apr'][i]=9
          if(data['month_sep'][i]):
            data['month_apr'][i]=10
          if (data['day_of_week_mon'][i]):
            data['day_of_week_fri'][i]=2
          if(data['day_of_week_thu'][i]):
            data['day_of_week_fri'][i]=3
          if (data['day_of_week_tue'][i]):
            data['day_of_week_fri'][i]=4
          if(data['day_of_week_wed'][i]):
            data['day_of_week_fri'][i]=5

        data['default']=(data['default']+1)
        data['housing']=(data['housing']+1)
        data['loan']=(data['loan']+1)
        train_data, test_data = split_data(data)
        numeric_attrs = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", 
                         "loan",  "education", "default", "housing", "poutcome_failure",
                         "job_admin.","marital_divorced","contact_cellular","month_apr","day_of_week_fri"]
        

        trainfeatures = torch.tensor(np.array(train_data[numeric_attrs])).float()
        trainlabels = torch.tensor(np.array(train_data['y'])).long()
        testfeatures = torch.tensor(np.array(test_data[numeric_attrs])).float()
        testlabels = torch.tensor(np.array(test_data['y'])).long()

    # Sample data
    num_train_0 = sum(i == 0 for i in trainlabels)
    num_train_1 = sum(i == 1 for i in trainlabels)
    num_test_0 = sum(i == 0 for i in testlabels)
    num_test_1 = sum(i == 1 for i in testlabels)

    a = int(num_train_0/num_train_1)
    b = int(num_test_0/num_test_1)
    dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
    weights_train = [a if label == 1 else 1 for data, label in dataset_train]
    sampler_train = WeightedRandomSampler(weights_train, num_samples=int(num_train_1)*2, replacement=False)
    index_train =copy.deepcopy(list(sampler_train))
    train_iter = Data.DataLoader(
      dataset=dataset_train,  
      batch_size=batch_size,  
      sampler=index_train,
      drop_last=True,
      shuffle=False,
    )

    dataset_test = Data.TensorDataset(testfeatures, testlabels)
    weights_test = [b if label == 1 else 1 for data, label in dataset_test]
    sampler_test = WeightedRandomSampler(weights_test, num_samples=int(num_test_1)*2, replacement=False)
    index_test = copy.deepcopy(list(sampler_test))
    test_iter = Data.DataLoader(
      dataset=dataset_test, 
      batch_size=batch_size, 
      sampler=index_test,
      drop_last=True,
      shuffle=False,
    )
    size_train = len(sampler_train)
    size_test = len(sampler_test)

    return train_iter, test_iter, size_train, size_test