import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans 
import seaborn as sns
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import pickle

st.title('Customer Personality Analysis')

st.write("Analysis of company's ideal customers")

data = pd.read_csv("./marketing_campaign.csv", delimiter = '\t')
data['Age'] =  2023  - data['Year_Birth'] 
data.drop(['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

#removing one outlier in Income
index = data[ (data['Income'] == data['Income'].max())].index
data.drop(index, inplace=True)
data = data.drop(data[data.Age > 100].index)

#Figure 1 Histogram
fig = plt.figure(figsize=(10, 5))
sns.histplot(data, x=data["Age"])
st.pyplot(fig)
st.write("According to the data, most customers are around the age of 40 & 50 ")
st.markdown("""---""")

#Figure 2 Pie Chart
fig2 = plt.figure()
data_edu = data.groupby(['Education'])['Education'].count()
data_edu.plot(kind='pie',
                            figsize=(3, 3),
                            autopct='%1.1f%%', # add in percentages
                            startangle=90,     # start angle 90° (Africa)
                            )

plt.title('Total number of Customers by Education Status')
plt.axis('equal') # Sets the pie chart to look like a circle.
st.pyplot(fig2)
st.write("According to the pie chart, over half of the customers are graduated")
st.markdown("""---""")

# #Figure 3 Bar Chart
response_count = data["Response"].value_counts()
response_percentage = (response_count / response_count.sum()) * 100
bar_x=[str(x) for x in response_count.index.tolist()]
bar_height = response_count.values

fig3 = plt.figure(figsize = (10, 5))
plt.bar(bar_x,bar_height, color ='maroon',
        width = 0.4)
 
plt.xlabel("Response")
plt.ylabel("Percentage")
plt.title("Percentage of the Last Campaign")
plt.show()
st.pyplot(fig3)
st.write("By looking at the bar chart, it is obvious that the last campaign was a failure with a high rate of no response by the customers")
st.markdown("""---""")

#Figure 4 
df_age = pd.DataFrame()
list_age = [24, 30, 35, 40, 45, 50, 55, 60, 65, 70]
for i in range(len(list_age)):
    if(i==len(list_age)-1):
        feature_format = "{}+".format(list_age[i])
        df_age.loc["Response 0", feature_format] = int(data[(data["Age"]>=list_age[i])&(data["Response"]==0)].shape[0])
        df_age.loc["Response 1", feature_format] = int(data[(data["Age"]>=list_age[i])&(data["Response"]==1)].shape[0])
    else:
        feature_format = "{}-{}".format(list_age[i], list_age[i+1])
        df_age.loc["Response 0", feature_format] = int(data[(data["Age"]>=list_age[i])&(data["Age"]<list_age[i+1])&(data["Response"]==0)].shape[0])
        df_age.loc["Response 1", feature_format] = int(data[(data["Age"]>=list_age[i])&(data["Age"]<list_age[i+1])&(data["Response"]==1)].shape[0])
    
    df_age.loc["Total", feature_format] = df_age.loc["Response 0", feature_format] + df_age.loc["Response 1", feature_format]
    df_age.loc["Response_0_Percentage", feature_format] = (df_age.loc["Response 0", feature_format]/df_age.loc["Total", feature_format])*100
    df_age.loc["Response_1_Percentage", feature_format] = (df_age.loc["Response 1", feature_format]/df_age.loc["Total", feature_format])*100

df_age = df_age.T.reset_index()
df_age = df_age.rename(columns={"index": "Age"})

fig4 = plt.figure(figsize = (10, 5))

sns.barplot(x="Age",

           y="Response_1_Percentage",

           data=df_age)
fig4.show()
st.pyplot(fig4)
st.write("As the figure indicated, the last campaign was more popular in 24-30 compared to other groups. However, since the main customers are around 40 and 50, the company needs to improve or renovate new campaign to focus on those main groups.")
st.markdown("""---""")

#Predict with model

st.write("Enter other Customers' data to predict")
with st.form("predict_form"):
    input22 = st.text_input('Age')
    input1 = st.text_input("Income")
    input2 = st.text_input('Kidhome')
    input3 = st.text_input('Teenhome')
    input4 = st.text_input('Recency')
    input5 = st.text_input('MntWines')
    input6 = st.text_input('MntFruits')
    input7 = st.text_input('MntMeatProducts')
    input8 = st.text_input('MntFishProducts')
    input9 = st.text_input('MntSweetProducts')
    input10 = st.text_input('MntGoldProds')
    input11 = st.text_input('NumDealsPurchases')
    input12 = st.text_input('NumWebPurchases')
    input13 = st.text_input('NumCatalogPurchases')
    input14 = st.text_input('NumStorePurchases')
    input15 = st.text_input('NumWebVisitsMonth')
    input19 = st.text_input('AcceptedCmp1')
    input20 = st.text_input('AcceptedCmp2')
    input16 = st.text_input('AcceptedCmp3')
    input17 = st.text_input('AcceptedCmp4')
    input18 = st.text_input('AcceptedCmp5')
    input21 = st.text_input('Complain')

    unseendata=pd.DataFrame({
                            'Income':[input1],
                            'Kidhome':[input2],
                            'Teenhome':[input3],
                            'Recency':[input4],
                            'MntWines':[input5],
                            'MntFruits':[input6],
                            'MntMeatProducts':[input7],
                            'MntFishProducts':[input8],
                            'MntSweetProducts':[input9],
                            'MntGoldProds':[input10],
                            'NumDealsPurchases':[input11],
                            'NumWebPurchases':[input12],
                            'NumCatalogPurchases':[input13],
                            'NumStorePurchases':[input14],
                            'NumWebVisitsMonth':[input15],
                            'AcceptedCmp3':[input16],
                            'AcceptedCmp4':[input17],
                            'AcceptedCmp5':[input18],
                            'AcceptedCmp1':[input19],
                            'AcceptedCmp2':[input20],
                            'Complain':[input21],
                            'Age':[input22]
                            })
    submitted = st.form_submit_button("Predict")
    if submitted:
        filename='classmodel'
        newmodel = pickle.load(open(filename, "rb"))
        res= newmodel.predict(unseendata)
        st.write(res)