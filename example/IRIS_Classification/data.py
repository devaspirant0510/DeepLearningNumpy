from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_iris_data():
    # iris data 불러오기
    iris_data = datasets.load_iris()
    # target 값 가져오기
    data = iris_data.data
    target = iris_data.target
    target_names = iris_data.target_names
    feature_names = iris_data.feature_names
    # x y 값 concat
    iris_df_data = np.concatenate([data,target.reshape(-1,1)],axis=1)
    feature_names.append("ytarget")
    # 데이터 프레임으로 만들기
    iris_df = pd.DataFrame(iris_df_data,columns=feature_names)
    # 현재 경로
    pwd = os.getcwd()+"\\example\\IRIS_Classification\\"
    # 파일이름
    file_name = "dataset\\iris_data.csv"
    # 현재 경로에 dataset 폴더가 있는지 확인
    if "dataset" in os.listdir(pwd):
        path = pwd+file_name
        iris_df.to_csv(path)
    else:
        os.mkdir(pwd+"\\dataset")
        iris_df.to_csv(pwd+file_name)
        
def get_iris_dataframe():
    root_path = os.getcwd()+"\\example\\IRIS_Classification"
    dir_path = root_path+"\\dataset"
    file_path = dir_path+"\\iris_data.csv"
    if "dataset" in os.listdir(root_path):
        if "iris_data.csv" in os.listdir(dir_path):
            return pd.read_csv(file_path,index_col=0)
        else:
            save_iris_data()
            get_iris_dataframe()    
    else:
        save_iris_data()
        get_iris_dataframe()    


df = get_iris_dataframe()

sns.relplot(data = df,x=df.iloc[:,0],hue=df.iloc[:,4],y=df.iloc[:,1])
plt.show()