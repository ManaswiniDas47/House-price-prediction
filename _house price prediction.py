#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


boston=load_boston()


# In[5]:


print(boston.DESCR)


# # Gather data
# [soure orignal resurch paper](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

# ## Data points and features

# In[6]:


boston.keys()


# In[7]:


print(boston.feature_names)


# In[8]:


bs = pd.DataFrame(boston.data)


# In[9]:


bs.head()


# In[10]:


bs.columns=boston.feature_names
bs.head()


# In[11]:


boston .target.shape


# In[12]:


bs['PRICE']=boston.target


# In[13]:


bs.head()


# In[14]:


bs.tail()


# In[15]:


bs.count()


# ## cleaning data_check for missing values

# In[16]:


pd.isnull(bs).any()


# In[17]:


bs.info()


# ## visualising Data - Histograms Distributions and Bar charts

# In[18]:


plt.figure(figsize=(10,6))
plt.hist(bs['PRICE'],bins=50,ec='black',color='#DEBACE')
plt.xlabel('price in 000s')
plt.ylabel ('NR. of Houses')
plt.show()


# In[19]:


plt.figure(figsize=(10,6))
sns.distplot(bs['PRICE'],bins=50,hist=True,color="#332FD0")
plt.show()


# ## cleaning data- check for mising values

# In[20]:


bs['RM'].mean


# In[21]:


plt.figure(figsize=(10,6))
plt.hist(bs['RM'],bins=50,ec='red',color='#0E5E6F')
plt.xlabel('price in rupes')
plt.ylabel ('NR. of Rooms')
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
sns.distplot(bs["RM"],bins=50,hist=True,color='#FD841F')
plt.show()


# In[ ]:





# ## challanges ; create a meningful histogram for RAD using matplotlib.......in Royal purple

# In[23]:


plt.figure(figsize=(10,6))
plt.hist(bs['RAD'],bins=50,ec="black",color='#9A1663')
plt.xlabel('Acesibility to Highways')
plt.ylabel('Nr.of Houses')
plt.show()


# In[24]:


bs['RAD'].value_counts()


# In[25]:


freqency=bs['RAD'].value_counts()
#type(freqency)
#freqency.index
#freqency.axes[0]
plt.bar(freqency .index,height=freqency)

plt.show()


# In[26]:


plt.figure(figsize=(10,6))
plt.hist(bs['RAD'],bins=24,ec='black',color='#0E5E6F',rwidth=0.5)
plt.xlabel('price in rupes')
plt.ylabel ('NR. of Rooms')
plt.show()


# In[27]:


bs['CHAS'].value_counts()


# ## Descriptive statistics

# ## $$\rho_{xy}= corr(x,y)$$

# ## $$ -1.0\leq\rho_{xy}\leq + 1.0 $$

# In[28]:


bs['PRICE'].min()


# In[29]:


bs['PRICE'].max()


# In[30]:


bs['PRICE'].corr(bs['RM'])


# In[31]:


bs['PRICE'].corr(bs['PTRATIO'])


# In[32]:


bs.describe()


# In[33]:


bs.corr()


# In[34]:


mask = np.zeros_like(bs.corr())
tringle_indices=np.triu_indices_from(mask)
mask[tringle_indices]=True
mask


# In[35]:


plt.figure(figsize=(16,10))
sns.heatmap(bs.corr(),mask=mask,annot=True,annot_kws={"size":14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[36]:


#challange :picture the relationship between pollution and distance in your head
#then create a scatter plot between Dis and NOX.


# In[37]:


nox_dis_corr=  round(bs['NOX'].corr(bs['DIS']),3)
plt.scatter(x=bs['DIS'],y=bs['NOX'],alpha=0.6, s=80, color='Indigo' )
plt. xlabel('DIS_Distance from employment',fontsize=14)
plt.ylabel('NOX_Nitric oxide polution',fontsize=14)
plt.title(f'DIS Vs NOX(correlation{nox_dis_corr})',fontsize=14)
plt.figure(figsize=(9,6))
plt.show()


# In[38]:


sns.set()
sns.set_context('talk')
sns.jointplot(x=bs['DIS'], y=bs['NOX'],height=8,color='Indigo',joint_kws={'alpha':0.5})
sns.set_style('whitegrid')
plt.show()


# In[102]:


sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=bs['DIS'],y=bs['NOX'],kind='hex',height=7,color='sky')


# In[40]:


sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=bs['TAX'],y=bs['RAD'],height=7,color='darkred',joint_kws={'alpha':0.5})


# In[41]:


sns.lmplot(x='TAX',y='RAD',data=bs,height=7)
plt.show()


# In[ ]:





# In[42]:


#  nox_dis_corr=bs['NOX'],corr(bs['DIS'])
# plt.title(f'DIS VS NOX'(correlation{nox_dis_corr})',fontsize=14)
# plt.xlabel('Dis-Distance from employment',fontsize=14)
# plt.ylabel('NOX- Nitric oxide polution',fontsize=14)
rm_tgt_corr=round(bs['RM'].corr(bs['PRICE']),3)
plt.figure(figsize=(9,6))
plt.scatter(x=bs['RM'] ,y=bs['PRICE'],alpha=0.6,s=80,color='skyblue')
plt.title(f'RM VS PRICE(correlation{rm_tgt_corr})',fontsize=14)
plt.xlabel('RM-Meadian nr of rooms',fontsize=14)
plt.ylabel('PRICE-property Price in 000s',fontsize=14)
# plt.title(f'RM VS PRICE(correlation{rm_tgt_corr})',fontsize=14)
# plt.figure(figsize=(9,6))
plt.show()


# # now we can perform the seaborn 

# In[43]:


sns.lmplot(x='RM',y='PRICE',data=bs,height=7)
plt.show()


# In[44]:


get_ipython().run_cell_magic('time', '', '\nsns.pairplot(bs)\nplt.show()')


# In[45]:


get_ipython().run_cell_magic('time', '', "\nsns.pairplot(bs,kind='reg',plot_kws={'line_kws':{'color':'cyan'}})\nplt.show()")


# # Multiple linearRegression(Multivariable Regreession)

#  $ Training & Test Dataset Split $

# In[46]:


prices = bs['PRICE']
features = bs.drop('PRICE',axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)


#% training set

len(x_train)/len(features)


# In[47]:


#% of test dataset
x_test.shape[0]/features.shape[0]


# In[48]:


# print('training data r-squared:',reg.score(x_train,y_train))
# print('test data r_squard:',reg.score(x_test,y_test))


# In[49]:


#% of test data set
x_test.shape[0]/features.shape[0]


# In[50]:


regr =LinearRegression()
regr.fit(x_train,y_train)
print('Intercept',regr.intercept_)
pd.DataFrame(data=regr.coef_,index=x_train.columns,columns=['Coef'])


# ## challange: print out r-squared for training and test datasets.

# In[51]:


print('training data r-squared:',regr.score(x_train,y_train))
print('test data r_squard:',regr.score(x_test,y_test))


# In[52]:


print('training data r-squared:',regr.score(x_train,y_train))
print('test data r_squard:',regr.score(x_test,y_test))

print('Intercept',regr.intercept_)
pd.DataFrame(data=regr.coef_,index=x_train.columns,columns=['Coef'])


# In[53]:


x_test


# In[54]:


x_train


# In[55]:


y_test


# In[56]:


y_train


# In[57]:


bs['PRICE'].skew


# In[58]:


#from math import log


# In[59]:


y_log=np.log(bs["PRICE"])
y_log.head()


# In[60]:


y_log.tail()


# In[61]:


y_log.skew()


# In[62]:


sns.distplot(y_log)
plt.title(f'Log price with skew{y_log.skew()}')
plt.show()


# In[63]:


bs.describe()


# In[64]:


trasformed_bs=features
trasformed_bs['LOG_PRICE']=y_log
sns.lmplot(x='LSTAT',y='LOG_PRICE',data=trasformed_bs,height=7,scatter_kws={'alpha':0.6},line_kws={'color':'cyan'})
plt.show()


# # Regression using LOG PRICES

# In[65]:


prices = np.log(bs['PRICE']) #use to log price
features = bs.drop('PRICE',axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)

print('training data r-squared:',regr.score(x_train,y_train))
print('test data r_squard:',regr.score(x_test,y_test))

print('Intercept',regr.intercept_)
pd.DataFrame(data=regr.coef_,index=x_train.columns,columns=['Coef'])


# In[66]:


np.e**0.080474


# # Evaluating coeficients

# In[67]:


x_incl_const = sm.add_constant(x_train)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()


# In[68]:


#  results.params


# In[69]:


#results.pvalues


# In[70]:


pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})


# ## Testing for Multilinearity
# $$ TAX = \alpha_0+\alpha_1 RM +\alpha_2 NOX+........+\alpha_{12}LSTAT $$
# $$ VIF_{TAX}=\frac{1}{(1-R_{TAX}^2)}$$

# In[71]:


# variance_inflation_factor(exog=x_incl_const,exog_idx=1)


# In[72]:


# type(x_incl_const)


# In[73]:


variance_inflation_factor(exog=x_incl_const.values,exog_idx=1)


# In[74]:


# challange : print out the number of columns in x_incl_constant?


# In[75]:


len(x_incl_const.columns)


# In[76]:


x_incl_const.shape[1]


# In[77]:


#challange:write a for loop that prints out all the VIFS for all the features?
for i in range(x_incl_const.shape[1]):
    print(variance_inflation_factor(exog=x_incl_const.values,exog_idx=i))
print("All Done!")


# In[78]:


vif = [] #Empty list
for i in range(x_incl_const.shape[1]):
    vif.append(variance_inflation_factor(exog=x_incl_const.values,exog_idx=i))
print(vif)


# In[79]:


vif = [variance_inflation_factor(exog=x_incl_const.values,exog_idx=i)for i in range(x_incl_const.shape[1])] 
print(vif)


# In[80]:


pd.DataFrame({'coef_name':x_incl_const.columns,
             'vif':np.around(vif, 2)})


# # Baysian Information Certerion (BIC)

# In[81]:


# orignal model with log prices and all features

x_incl_const = sm.add_constant(x_train)

model=sm.OLS(y_train,x_incl_const)
results=model.fit()

org_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})
#challange: find and check official docs for results object and print out BIC & r_squared
print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[82]:


# Reduced model #1. excluding INDUS
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

coef_minus_indus = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[83]:


# Reduced model #1. excluding INDUS
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['CHAS'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

coef_minus_CHAS = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[84]:


# Reduced model #1. excluding INDUS
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['PTRATIO'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

coef_minus_PTRATIO = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[85]:


# Reduced model #1. excluding INDUS
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['B'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

coef_minus_B = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[86]:


# Reduced model #1. excluding INDUS anad AGE
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS','AGE'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

reduce_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[87]:


# Reduced model #1. excluding INDUS anad LSTAT
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS','LSTAT'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

reduce_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[88]:


# Reduced model #1. excluding INDUS anad RM
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS','RM'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

reduce_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[89]:


# Reduced model #1. excluding INDUS and CRIM
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS','CRIM'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

reduce_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[ ]:





# In[90]:


# Reduced model #1. excluding INDUS anad ZN
x_incl_const = sm.add_constant(x_train)
x_incl_const = x_incl_const.drop(['INDUS','ZN'],axis=1)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

reduced_coef = pd.DataFrame({'coef':results.params,'p_values':round(results.pvalues,3)})

print('BIC is',results.bic)
print('rsquared is',results.rsquared)


# In[91]:


frames=[org_coef,coef_minus_indus,reduced_coef]
pd.concat(frames, axis=1)


# ## rediduals and residuals plots

# In[92]:


# Modified model:transformed (using log prices) & simplified (droping two features)
prices = np.log(bs['PRICE']) #use to log price
features = bs.drop(['PRICE' ,'INDUS','AGE'],axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)

# using statsmodel

x_incl_const= sm.add_constant(x_train)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

# Residuals

# residuals =y_train- results.fittedvalues
# residuals.describe()
# results.resid

# Graph of Actual vs. predicted prices

corr =round(y_train.corr(results.fittedvalues),2)
corr
plt.scatter(x=y_train, y=results.fittedvalues , c='navy',alpha=0.6)
plt.plot(y_train,y_train, color='cyan')
plt.xlabel('Actual log prices $y _i$',fontsize=14)
plt.ylabel('Predicted log price $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted log prices:$y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()

plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues , c='blue',alpha=0.6)
plt.plot(np.e**y_train,np.e**y_train, color='cyan')
plt.xlabel('Actual prices 0000s $y _i$',fontsize=14)
plt.ylabel('Predicted price 0000s $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted  prices:$y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()

# Resudual vs Predicted Values
plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues , c='blue',alpha=0.6)
plt.plot(np.e**y_train,np.e**y_train, color='cyan')
plt.xlabel('Actual prices 0000s $y _i$',fontsize=14)
plt.ylabel('Predicted price 0000s $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted  prices:$y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()


# Resudual vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid , c='navy',alpha=0.6)

plt.xlabel('predicted log prices $\hat y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.title('Resudual vs fitted values',fontsize=17)
plt.show()


# In[93]:


# Resudual vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid , c='navy',alpha=0.6)

plt.xlabel('predicted log prices $\hat y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.title('Resudual vs fitted values',fontsize=17)
plt.show() 


# In[94]:


# Distribution of Residuals (log prices) - checking for normality
resid_mean = round(results.resid.skew(),3)
resid_skew = round(results.resid.skew(),3)

sns.distplot(results.resid, color='blue')
plt.title(f'Log price Model:residuals skew ({resid_skew})Mean({resid_skew})')
plt.show()


# In[95]:


# challange: Using orignal model with all the features and normal prices generate:
# plot of actual vs predicted prices(incl.correlation) using different color
# plot of residuals vs predicted prices
# plot of Distribution of residuals(incl.skew)
# Analyse the results.


# In[96]:


# orignal model:normal prices & all features
prices = bs['PRICE']
features = bs.drop(['PRICE'],axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)


x_incl_const= sm.add_constant(x_train)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

# Graph of Actual vs. predicted prices

corr =round(y_train.corr(results.fittedvalues),2)
plt.scatter(x=y_train, y=results.fittedvalues , c='indigo',alpha=0.6)
plt.plot(y_train,y_train, color='cyan')
plt.xlabel('Actual prices 000s $y _i$',fontsize=14)
plt.ylabel('Predicted price 000s $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted prices:$y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()

# Resudual vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid , c='indigo',alpha=0.6)

plt.xlabel('predicted prices $\hat y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.title('Resudual vs fitted values',fontsize=17)
plt.show()

# Resudual Distribution chart

resid_mean = round(results.resid.skew(),3)
resid_skew = round(results.resid.skew(),3)

sns.distplot(results.resid, color='indigo')
plt.title(f'Residuals skew ({resid_skew})Mean({resid_skew})')
plt.show()

# Mean Squared Error & Rsquared
full_normal_mse = round(results.mse_resid , 3)
full_normal_rsquared = round(results.rsquared , 3)


# In[97]:


# model omitting key features using log prices
prices =np.log (bs['PRICE'])
features = bs.drop(['PRICE','INDUS','AGE','LSTAT','RM','NOX','CRIM'],axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)


x_incl_const= sm.add_constant(x_train)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

# Graph of Actual vs. predicted prices

corr =round(y_train.corr(results.fittedvalues),2)
plt.scatter(x=y_train, y=results.fittedvalues , c='#e74c3c',alpha=0.6)
plt.plot(y_train,y_train, color='cyan')
plt.xlabel('Actual log prices $y _i$',fontsize=14)
plt.ylabel('Predicted log price $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted prices with omitted variables: $y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()

# Resudual vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid , c='#e74c3c',alpha=0.6)

plt.xlabel('predicted prices $\hat y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.title('Residuals vs fitted values',fontsize=17)
plt.show()

# Mean Squared Error & Rsquared
omitted_var_mse = round(results.mse_resid , 3)
omitted_var_rsquared = round(results.rsquared , 3)


# In[ ]:





# In[98]:


# modified model: transformed (using log prices) & simplified (dropping two feature)
prices = bs['PRICE']
features = bs.drop(['PRICE'],axis=1)

x_train, x_test, y_train, y_test= train_test_split(features, prices, test_size=0.2, random_state=10)


x_incl_const= sm.add_constant(x_train)
model=sm.OLS(y_train,x_incl_const)
results=model.fit()

# Graph of Actual vs. predicted prices

corr =round(y_train.corr(results.fittedvalues),2)
plt.scatter(x=y_train, y=results.fittedvalues , c='indigo',alpha=0.6)
plt.plot(y_train,y_train, color='cyan')
plt.xlabel('Actual prices 000s $y _i$',fontsize=14)
plt.ylabel('Predicted price 000s $\hat y _i$',fontsize=14)
plt.title(f'Actual vs predicted prices:$y_i$ vs $\hat y _i$ (corr{corr})',fontsize=17)
plt.show()

# Resudual vs Predicted Values
plt.scatter(x=results.fittedvalues, y=results.resid , c='indigo',alpha=0.6)

plt.xlabel('predicted prices $\hat y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.title('Resudual vs fitted values',fontsize=17)
plt.show()

# Mean Squared Error & Rsquared
reduced_log_mse = round(results.mse_resid , 3)
reduced_log_rsquared = round(results.rsquared , 3)


# In[99]:


pd.DataFrame({'R-Squared':[reduced_log_rsquared,full_normal_rsquared,omitted_var_rsquared],
             'MSE': [reduced_log_mse,full_normal_mse,omitted_var_mse],
             'RMSE':np.sqrt([reduced_log_mse,full_normal_mse,omitted_var_mse])},
            index=['Reduced Log Model' , 'Full Normal Price Model' ,'Omitted var Model'])


# In[100]:


# challange: our Estimate for house price is $30,000. calculate teh upper and lower bound
# For a 95% prediction interval using a log model

print('1 s.d in log prices',np.sqrt(reduced_log_mse))
print('2 s.d in log prices' ,2*np.sqrt(reduced_log_mse))
# print('4 s.d in log prices',2**np.sqrt(reduced_log_mse))

upper_bound = np.log(30) + 2*np.sqrt(reduced_log_mse)
print('The upper bound in log prices for a 95% prediction interval is' ,upper_bound)
print('The upper bound in normal prices is $' ,np.e**upper_bound*1000)
lower_bound = np.log(30) - 2*np.sqrt(reduced_log_mse)
print('The lower bound in log prices for a 95% prediction interval is' ,lower_bound)
print('The lower bound in normal prices is $' , np.e**lower_bound*1000)


# In[101]:


30000 + np.e**(2*np.sqrt(reduced_log_mse))*100 # wrong add first.Transform afterwards.


# In[ ]:




