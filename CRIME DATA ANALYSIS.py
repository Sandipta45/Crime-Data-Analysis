#!/usr/bin/env python
# coding: utf-8

# # Installing Important Libraties

# In[1]:


#!pip install wordcloud
#!pip install folium


# # Importing neccessary libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# # Reading the CSV file

# In[3]:


df=pd.read_csv("police_department_data.csv")
df.head(2)


# In[4]:


df.shape


# In[5]:


df.sample(2)


# In[6]:


df.info()


# In[7]:


# Here, we finding any NA Values in any cell.
df.isna().sum()


# In[8]:


df.duplicated(["incident_id"]).sum()


# In[9]:


df.drop_duplicates(subset ="incident_id", 
                     keep = False, inplace = True) 


# In[10]:


df['department_district'].fillna(df['department_district'].mode()[0], inplace=True)


# In[11]:


df["crime_date"]= pd.to_datetime(df["crime_date"]) 


# In[12]:


df['crime_date'] = pd.to_datetime(df['crime_date'])
df['Date'] = df['crime_date'].dt.strftime('%d/%m/%Y')
df['Time'] = df['crime_date'].dt.strftime('%H:%M')


# In[13]:


df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day


# In[14]:


#df["Date"]= pd.to_datetime(df["Date"])
#df["Time"]= pd.to_datetime(df["Time"])
df["year"]=df["year"].astype(int)
df["month"]=df["month"].astype(int)
df["Day"]=df["Day"].astype(int)


# In[15]:


df.head(2)


# In[16]:


df[['latitude','longitude']] = df['location'].str.split(expand=True,n=1,)
df['latitude'] = df['latitude'].str[1:]
df['latitude'] = df['latitude'].str[:-1]
df['longitude'] = df['longitude'].str[:-1]
df['latitude'] = df['latitude'].astype(float)
df['longitude'] =df['longitude'].astype(float)
df.sample(2)


#                         # Number of crimes category wise

# In[17]:


df['category'].value_counts().sort_index()


# In[18]:


# Creating the graph on the basis of Days
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
dfCrimeYear = pd.pivot_table(df, values=["category"],index = ["Day"], aggfunc='count')
f, ax = plt.subplots(1,1, figsize = (12, 4), dpi=100)
xdata = dfCrimeYear.index
ydata = dfCrimeYear
ax.plot(xdata, ydata)
ax.set_ylim(ymin=0, ymax=10000)
plt.xlabel('Day')
plt.ylabel('Number of Crimes')
plt.title('Crime in 2016')
plt.show()


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
dfCrimeType = pd.pivot_table(df, values=["month"],index = ["category"], aggfunc='count')
dfCrimeType = dfCrimeType.sort_values(["month"],ascending=True)
crimeplot = dfCrimeType.plot(kind='bar',
               figsize = (20,15),
               title='Number of Crimes Committed by Type'
             )

plt.rcParams["figure.dpi"] = 1000
plt.legend(loc='lower right')
plt.ylabel('Crime Type')
plt.xlabel('Number of Crimes')
plt.show(crimeplot)


# In[20]:


x = df.category
y = df.department_district

fig = go.Figure(go.Histogram2d(
        x=x,
        y=y
    ))
fig.show()


# In[21]:


labels = df.department_district.unique()
values=[]
for each in labels:
    values.append(len(df[df.department_district==each]))

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# In[22]:


# Create a pivot table with day and month; another that counts the number of years that each day had; and the average. 
crimes_pivot_table = df.pivot_table(values='year', index='Day', columns='month', aggfunc=len)
crimes_pivot_table_year_count = df.pivot_table(values='year', index='Day', columns='month', aggfunc=lambda x: len(x.unique()))
crimes_average = crimes_pivot_table/crimes_pivot_table_year_count
crimes_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.figure(figsize=(10,10))
plt.title('Average Number of Crime per Day and Month', fontsize=12)
sns.heatmap(crimes_average.round(), cmap='seismic', linecolor='green',linewidths=0.2,annot=True, fmt=".0f");


# In[23]:


import folium
from folium.features import CustomIcon
from folium.plugins import MarkerCluster


# In[24]:


plt.figure(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.category))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()


# In[25]:


df = pd.crosstab(df['category'], df['department_district'])
color = plt.cm.Greys(np.linspace(0, 1, 10))
df.div(df.sum(1).astype(float), axis = 0).plot.bar(stacked = True, color = color, figsize = (18, 12))
plt.title('District vs Category of Crime', fontweight = 30, fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:




