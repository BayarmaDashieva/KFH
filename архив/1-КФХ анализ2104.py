#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set (style= 'ticks')
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import  mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


# In[2]:


KFH_dataset =pd.read_excel ('1-КФХ.xlsx')


# In[3]:


pd.options.display.float_format = '{:.1f}'.format


# ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАТАСЕТА

# In[4]:


KFH_dataset.head ()


# In[5]:


KFH_dataset.shape


# In[6]:


KFH_dataset.dtypes


# In[7]:


KFH_dataset.describe ()


# ПОИСК ПРОПУСКОВ И ЗАПОЛНЕНИЕ ПРОПУСКОВ НУЛЯМИ

# In[8]:


KFH_dataset.isnull (). sum ()


# In[9]:


KFH_dataset.isna()


# In[10]:


KFH_dataset.isna().sum()


# In[11]:


df = KFH_dataset.fillna(0) # Заполнение всех пропущенных значений нулями
df.head()


# РАСЧЕТ ОТНОСИТЕЛЬНЫХ ВЕЛИЧИН И СОЗДАНИЕ ДАТАФРЕЙМА

# In[12]:


dohody_rub = df.iloc[:, 1] # Столбец Доходы
#subsidii = df.iloc[:, 4] # Столбец Субсидии
#rashody = df.iloc[:, 5] # Столбец Расходы
#mater_zatraty = df.iloc[:, 10] # Столбец Материальные затраты
#semena = df.iloc[:, 11] # Столбец Расходы на семена и посадочный материал 
#min_udobr = df.iloc[:, 12] # Столбец Расходы на минеральные удобрения, бактериальные и другие препараты
#zashita_rast = df.iloc[:, 13] # Столбец Расходы на средства защиты растений
rabotniki = df.iloc[:, 16] # Столбец Численность постоянных работников, чел
chleny_KFH = df.iloc[:, 17] # Столбец Члены КФХ, чел
#kredity = df.iloc[:, 19] # Столбец Долгосрочные кредиты, руб.
#zaimy = df.iloc[:, 21] # Столбец Долгосрочные займы, руб.
#posevy = df.iloc[:, 22] # Столбец Площадь посеянная, руб.
#proizvedeno_zerna =df.iloc[:, 24] # Столбец произведено зерна, ц
urozhainost =df.iloc[:, 25] # Столбец урожайность зерновых, ц/га
#realizovano_zerna = df.iloc[:, 26] # Столбец реализовано зерна, ц
#sxtechnics_nachalo = df.iloc[:, 46] # Столбец Сельскохозяйственная техника - всего, шт: - наличие на начало года
#sxtechnics_konec = df.iloc[:, 47] # Столбец Сельскохозяйственная техника - всего, шт: - наличие на конец года
tractory_nachalo = df.iloc[:, 48] # Столбец тракторы - наличие на начало года 
tractory_konec = df.iloc[:, 49] # Столбец тракторы - наличие на конец года 
combain_nachalo = df.iloc[:, 50] # Столбец комбайны - наличие на начало года 
combain_konec = df.iloc[:, 51] # Столбец комбайны - наличие на конец года 
land_nachalo = df.iloc[:, 52] # Столбец Земельные участки - всего, га - наличие на начало года
land_konec = df.iloc[:, 53] # Столбец Земельные участки - всего, га - наличие на конец года

#vsd= dohody - rashody  #новый столбец Валовые смешанные доходы

#sxtechnics_mean = (sxtechnics_nachalo + sxtechnics_konec)/ 2 # новый столбец Земельные участки, га - среднегодовая


#dohody_na_100ga = dohody / land_mean / 1000 *100 # расчет: доходы на 100 га сельхозугодий, тыс.руб.
#vsd_na_1rab_chl = vsd / chislennost_rab_chl / 1000 # расчет: валовые смешанные доходы на 1 работника и члена КФХ, тыс.руб.
#obesp_rs = chislennost_rab_chl / land_mean *100 # расчет: обеспеченность рабочей силой, чел. на 100 га сху
#mater_zatraty_na100rub_doh = mater_zatraty / dohody *100 # расчет: Материальные затраты на 100 руб. доходов
#subsidii_na100ga = subsidii / land_mean *100 /1000
#kredity_na100ga = kredity / land_mean *100 /1000
#zaimy_na100ga = zaimy / land_mean *100 /1000
#semena_na100ga =  semena / land_mean *100 / 1000
#min_udobr_na100ga =  min_udobr / land_mean *100 / 1000
#zashita_rast_na100ga =  zashita_rast / land_mean *100 / 1000
#tractory_na100ga = tractory_mean / land_mean *100
#combainy_na100ga = combainy_mean / land_mean *100
#posevnaya_ploshad = posevy
#tovarnost = realizovano_zerna / proizvedeno_zerna *100 # расчет: уровень товарности, %

chislennost_rab_chl = rabotniki + chleny_KFH # новый столбец Численность работников и членов КФХ
land_mean = (land_nachalo + land_konec) / 2 # новый столбец Земельные участки, га - среднегодовая
tractory_mean = (tractory_nachalo + tractory_konec)/ 2
combainy_mean = (combain_nachalo + combain_konec)/ 2 
dohody = dohody_rub / 1000 # расчет: доходы на 1 КФХ, тыс.руб.
rabotniki = chislennost_rab_chl  # расчет: постоянные работники и члены КФХ на 1 КФХ, чел.
tractory = tractory_mean
combainy = combainy_mean
zemlya = land_mean
urozhainost_zerna = urozhainost


# In[13]:


df_pok = pd.DataFrame()                   # Создание пустого датафрейма 
#df_pok ['Доходы на 100 га сельхозугодий, тыс.руб.'] = dohody_na_100ga # Добавление столбца 'Доходы на 100 га сельхозугодий, тыс.руб.'
#df_pok ['ВСД на 1 работника и члена КФХ, тыс.руб.'] = vsd_na_1rab_chl # Добавление столбца 'Валовые смешанные доходы на 1 работника и члена КФХ, тыс.руб.'
#df_pok ['Обеспеченность рабочей силой, чел. на 100 га сху'] = obesp_rs  # Добавление столбца 'Обеспеченность рабочей силой, чел. 
#df_pok ['Материальные затраты на 100 руб. доходов, руб.'] = mater_zatraty_na100rub_doh # Добавление столбца 'Материальные затраты на 100 руб. доходов, руб.'
#df_pok ['Субсидии на 100 га сху, тыс.руб.'] =  subsidii_na100ga # Добавление столбца 'Субсидии на 100 га сху, тыс. руб.'
#df_pok ['Кредиты на 100 га сху, тыс.руб.'] =  kredity_na100ga # Добавление столбца 'Кредиты на 100 га сху, тыс. руб.'
#df_pok ['Займы на 100 га сху, тыс.руб.'] =  zaimy_na100ga # Добавление столбца 'Займы на 100 га сху, тыс. руб.'
#df_pok ['Расходы на семена и посадочный материал, тыс. руб. на 100 га сху'] =  semena_na100ga # Добавление столбца 'Расходы на семена и посадочный материал, тыс. руб. на 100 га сху'
#df_pok ['Расходы на минеральные удобрения, тыс. руб. на 100 га сху'] =  min_udobr_na100ga # Добавление столбца 'Расходы на минеральные удобрения, тыс. руб. на 100 га сху'
#df_pok ['Расходы на средства защиты растений, тыс. руб. на 100 га сху'] =  zashita_rast_na100ga # Добавление столбца 'Расходы на средства защиты растений, тыс. руб. на 100 га сху'
#df_pok ['Наличие тракторов на 100 га сху, шт.'] = tractory_na100ga # Добавление столбца 'Наличие тракторов на 100 га сху, шт.'
#df_pok ['Наличие комбайнов на 100 га сху, шт.'] = combainy_na100ga # Добавление столбца 'Наличие комбайнов на 100 га сху, шт.'
#df_pok ['Выручка от реализ. зерновых на 1 га посевов, тыс.руб.'] = df1 ['Зерновые и зернобобовые культуры на зерно и семена (кроме риса) - доход от реализации, руб'] / df1 ['Зерновые и зернобобовые культуры на зерно и семена (кроме риса) - посеянная площадь, га']/ 1000


df_pok ['Доходы, тыс.руб.'] = dohody # Добавление столбца 'Доходы на 1 КФХ, тыс.руб.'
df_pok ['Работники, чел.'] = rabotniki # Добавление столбца 'Работники на 1 КФХ, тыс.руб.'
df_pok ['Наличие тракторов, шт.'] = tractory # Добавление столбца 'Наличие тракторов, шт.'
df_pok ['Наличие комбайнов, шт.'] = combainy # Добавление столбца 'Наличие комбайнов, шт.'
df_pok ['Общая площадь земли, га'] = zemlya # Добавление столбца 'Общая площадь земли, га'
df_pok ['Урожайность зерновых, га'] = urozhainost_zerna # Добавление столбца 'Урожайность зерновых, га'

#df_pok ['Уровень товарности зерновых, %'] = tovarnost # Добавление столбца 'Уровень товарности зерновых, %'
#df_pok ['Посевная площадь зерновых, га'] = posevnaya_ploshad # Добавление столбца 'Посевная площадь зерновых, га'
#df_pok ['Субсидии на 1 КФХ, тыс.руб.'] =  subsidii / 1000 # Добавление столбца 'Субсидии на 1 КФХ, тыс. руб.'
#df_pok ['Кредиты на 1 КФХ, тыс.руб.'] = df1 ['долгосрочные (более 1 года) - получено \nза год'] / 1000 
#df_pok ['Займы на 1 КФХ, тыс.руб.'] = df1 ['долгосрочные (более 1 года) - получено \nза год.1'] / 1000 


# In[14]:


#df_pok ['Численность постоянных работников, чел'] = rabotniki # Добавление столбца 'Численность постоянных работников, чел'
#df_pok ['Численность постоянных работников, чел'].mask ( df_pok ['Численность постоянных работников, чел'] != 0, 'да')
#df_pok ['Численность постоянных работников, чел'].mask ( df_pok ['Численность постоянных работников, чел'] == 0, 'нет')
#df_pok1 = df_pok.rename(columns={'Численность постоянных работников, чел': 'Найм работников'})


# РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ

# In[15]:


df_pok.head()


# In[16]:


df_pok.shape


# In[17]:


df_pok.dtypes


# In[18]:


df_pok.info ()


# In[19]:


df_pok.describe ()


# In[20]:


df_pok.isnull (). sum ()


# ВИЗУАЛЬНОЕ ИССЛЕДОВАНИЕ ДАТАСЕТА

# In[21]:


sns.pairplot (df_pok) # построение попарных графиков рассеяния для всего набора данных


# In[22]:


sns.pairplot(df_pok, hue="Доходы, тыс.руб.")


# In[23]:


pd.options.display.float_format = '{:.3f}'.format
df_pok.corr () # изучение корреляции признаков


# In[24]:


df_pok.corr(method='pearson')


# In[25]:


df_pok.corr(method='kendall')


# In[26]:


df_pok.corr(method='spearman')


# In[27]:


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(20,10))
fig.suptitle('Корреляционная матрица:Pearson ')
sns.heatmap(df_pok.corr(method='pearson'), annot=True, fmt='.3f')


# In[28]:


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(20,10))
fig.suptitle('Корреляционная матрица:Kendall ')
sns.heatmap(df_pok.corr(method='kendall'), annot=True, fmt='.3f')


# In[29]:


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(20,10))
fig.suptitle('Корреляционная матрица:Spearman ')
sns.heatmap(df_pok.corr(method='spearman'), cmap='YlGnBu', annot=True, fmt='.3f')


# In[30]:


df_pok.hist(figsize=(20,20))
plt.show()


# ПРЕДОБРАБОТКА ДАННЫХ

# НОРМАЛИЗАЦИЯ 

# In[31]:


df_pok.head()


# In[32]:


def diagnostic_plots(df, variable, title):
    """
    df
    
    """
    fig, ax = plt.subplots(figsize=(15,6))
    # гистограмма
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=60)
    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    fig.suptitle(title)
    plt.show()


# In[33]:


diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - original')


# In[34]:


#df_pok ['Доходы, тыс.руб.'] = np.log(df_pok ['Доходы, тыс.руб.'])
#diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. логарифмическое преобразование')


# In[35]:


#df_pok ['Доходы, тыс.руб.'] = 1 / (df_pok ['Доходы, тыс.руб.']) 
#diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - обратное преобразование')


# In[36]:


#df_pok ['Доходы, тыс.руб.'] = (df_pok ['Доходы, тыс.руб.'])** (1/2)
#diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - корень квадратный')


# In[37]:


#df_pok ['Доходы, тыс.руб.'] = (df_pok  ['Доходы, тыс.руб.']) ** (2)
#diagnostic_plots(df_pok , 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - возведение в степень')


# In[38]:


#df_pok ['Доходы, тыс.руб.'], param = stats.boxcox(df_pok ['Доходы, тыс.руб.']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - преобразование Бокса-Кокса')


# In[39]:


df_pok ['Доходы, тыс.руб.'] = df_pok ['Доходы, тыс.руб.'].astype('float')
df_pok ['Доходы, тыс.руб.'], param = stats.yeojohnson(df_pok ['Доходы, тыс.руб.']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - преобразование Йео-Джонсона')


# In[40]:


diagnostic_plots(df_pok, 'Работники, чел.', 'Работники, чел. - original')


# In[41]:


#df_pok ['Работники, чел.'], param = stats.boxcox(df_pok ['Работники, чел.']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Работники, чел.', 'Работники, чел. - преобразование Бокса-Кокса')


# In[42]:


df_pok ['Работники, чел.'] = df_pok ['Работники, чел.'].astype('float')
df_pok ['Работники, чел.'], param = stats.yeojohnson(df_pok ['Работники, чел.']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Работники, чел.', 'Работники, чел. - преобразование Йео-Джонсона')


# In[43]:


diagnostic_plots(df_pok, 'Наличие тракторов, шт.', 'Наличие тракторов, шт.- original')


# In[44]:


#df_pok ['Наличие тракторов, шт.'], param = stats.boxcox(df_pok  ['Наличие тракторов, шт.']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Наличие тракторов, шт.', 'Наличие тракторов, шт. - преобразование Бокса-Кокса')


# In[45]:


df_pok ['Наличие тракторов, шт.'] = df_pok ['Наличие тракторов, шт.'].astype('float')
df_pok ['Наличие тракторов, шт.'], param = stats.yeojohnson(df_pok ['Наличие тракторов, шт.']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Наличие тракторов, шт.', 'Наличие тракторов, шт. - преобразование Йео-Джонсона')


# In[46]:


diagnostic_plots(df_pok, 'Наличие комбайнов, шт.', 'Наличие комбайнов, шт. - original')


# In[47]:


#df_pok ['Наличие комбайнов, шт.'], param = stats.boxcox(df_pok  ['Наличие комбайнов, шт.']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Наличие комбайнов, шт.', 'Наличие комбайнов, шт. - преобразование Бокса-Кокса')


# In[48]:


df_pok ['Наличие комбайнов, шт.'] = df_pok ['Наличие комбайнов, шт.'].astype('float')
df_pok ['Наличие комбайнов, шт.'], param = stats.yeojohnson(df_pok ['Наличие комбайнов, шт.']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Наличие комбайнов, шт.', 'Наличие комбайнов, шт. - преобразование Йео-Джонсона')


# In[49]:


#diagnostic_plots(df_pok, 'Посевная площадь зерновых, га', 'Посевная площадь зерновых, га - original')


# In[50]:


#df_pok ['Посевная площадь зерновых, га'], param = stats.boxcox(df_pok  ['Посевная площадь зерновых, га']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Посевная площадь зерновых, га', 'Посевная площадь зерновых, га - преобразование Бокса-Кокса')


# In[51]:


#df_pok ['Посевная площадь зерновых, га'] = df_pok ['Посевная площадь зерновых, га'].astype('float')
#df_pok ['Посевная площадь зерновых, га'], param = stats.yeojohnson(df_pok ['Посевная площадь зерновых, га']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Посевная площадь зерновых, га', 'Посевная площадь зерновых, га - преобразование Йео-Джонсона')


# In[52]:


diagnostic_plots(df_pok, 'Общая площадь земли, га', 'Общая площадь земли, га - original')


# In[53]:


#df_pok ['Общая площадь земли, га'], param = stats.boxcox(df_pok  ['Общая площадь земли, га']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Общая площадь земли, га', 'Общая площадь земли, га - преобразование Бокса-Кокса')


# In[54]:


df_pok ['Общая площадь земли, га'] = df_pok ['Общая площадь земли, га'].astype('float')
df_pok ['Общая площадь земли, га'], param = stats.yeojohnson(df_pok ['Общая площадь земли, га']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Общая площадь земли, га', 'Общая площадь земли, га - преобразование Йео-Джонсона')


# In[55]:


diagnostic_plots(df_pok, 'Урожайность зерновых, га', 'Урожайность зерновых, га- original')


# In[56]:


#df_pok ['Урожайность зерновых, га'], param = stats.boxcox(df_pok  ['Урожайность зерновых, га']) 
#print('Оптимальное значение λ = {}'.format(param))
#diagnostic_plots(df_pok, 'Урожайность зерновых, га', 'Урожайность зерновых, га - преобразование Бокса-Кокса')


# In[57]:


df_pok ['Урожайность зерновых, га'] = df_pok ['Урожайность зерновых, га'].astype('float')
df_pok ['Урожайность зерновых, га'], param = stats.yeojohnson(df_pok ['Урожайность зерновых, га']) 
print('Оптимальное значение λ = {}'.format(param))
diagnostic_plots(df_pok, 'Урожайность зерновых, га', 'Урожайность зерновых, га- преобразование Йео-Джонсона')


# In[58]:


df_pok.head()


# ОБРАБОТКА ВЫБРОСОВ

# In[59]:


def diagnostic_plots(df, variable, title):
    fig, ax = plt.subplots(figsize=(10,6))
    # гистограмма
    plt.subplot(2, 2, 1)
    df[variable].hist(bins=50)
    # Q-Q plot
    plt.subplot(2, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    # скрипичная диаграмма
    plt.subplot(2, 2, 3)
    sns.violinplot(x=df[variable])    
    # ящик с усами
    plt.subplot(2, 2, 4)
    sns.boxplot(x=df[variable])  
    fig.suptitle(title)
    plt.show()


# In[60]:


df_pok_out = df_pok.copy ()
df_pok_out.head()


# In[61]:


diagnostic_plots(df_pok_out, 'Доходы, тыс.руб.', 'Доходы, тыс.руб. - normal original')


# In[62]:


diagnostic_plots(df_pok_out, 'Работники, чел.', 'Работники, чел. - normal original')


# In[63]:


diagnostic_plots(df_pok_out, 'Наличие тракторов, шт.', 'Наличие тракторов, шт. - normal original')


# In[64]:


diagnostic_plots(df_pok_out, 'Наличие комбайнов, шт.', 'Наличие комбайнов, шт. - normal original')


# In[65]:


#diagnostic_plots(df_pok_out, 'Посевная площадь зерновых, га', 'Посевная площадь зерновых, га - normal original')


# In[66]:


diagnostic_plots(df_pok_out, 'Общая площадь земли, га', 'Общая площадь земли, га - normal original')


# In[67]:


diagnostic_plots(df_pok_out, 'Урожайность зерновых, га', 'Урожайность зерновых, га - normal original')


# In[68]:


# Тип вычисления верхней и нижней границы выбросов
from enum import Enum
class OutlierBoundaryType(Enum):
    SIGMA = 1
    QUANTILE = 2
    IRQ = 3


# In[69]:


# Функция вычисления верхней и нижней границы выбросов
def get_outlier_boundaries(df, col, outlier_boundary_type: OutlierBoundaryType):
    if outlier_boundary_type == OutlierBoundaryType.SIGMA:
        K1 = 3
        lower_boundary = df[col].mean() - (K1 * df[col].std())
        upper_boundary = df[col].mean() + (K1 * df[col].std())

    elif outlier_boundary_type == OutlierBoundaryType.QUANTILE:
        lower_boundary = df[col].quantile(0.05)
        upper_boundary = df[col].quantile(0.95)

    elif outlier_boundary_type == OutlierBoundaryType.IRQ:
        K2 = 1.5
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        lower_boundary = df[col].quantile(0.25) - (K2 * IQR)
        upper_boundary = df[col].quantile(0.75) + (K2 * IQR)

    else:
        raise NameError('Unknown Outlier Boundary Type')
        
    return lower_boundary, upper_boundary   


# In[70]:


df_pok_out.shape


# In[71]:


for col in df_pok_out:    
    for obt in OutlierBoundaryType:
        # Вычисление верхней и нижней границы
        lower_boundary, upper_boundary = get_outlier_boundaries(df_pok_out, col, obt)
        # Флаги для удаления выбросов
        outliers_temp = np.where(df_pok_out [col] > upper_boundary, True, 
                        np.where(df_pok_out [col] < lower_boundary, True, False))
        # Удаление данных на основе флага
        df_pok_out_trimmed = df_pok_out.loc[~(outliers_temp), ]  
        title = 'Поле-{}, метод-{}, строк-{}'.format(col, obt, df_pok_out_trimmed.shape[0])
        diagnostic_plots(df_pok_out_trimmed, col, title)


# In[72]:


df_pok_out_trimmed.shape


# In[73]:


df_pok_out_trimmed.head()


# In[74]:


df_pok_out_trimmed.describe ()


# In[75]:


df_pok_out_trimmed1 = df_pok_out_trimmed.copy ()


# In[76]:


# Функция для восстановления датафрейма на основе масштабированных данных
def arr_to_df(arr_scaled):
    res = pd.DataFrame(arr_scaled, columns=df_pok_out_trimmed1.columns)
    return res


# In[77]:


# Обучаем StandardScaler на всей выборке и масштабируем
cs11 = StandardScaler()
data_cs11_scaled_temp = cs11.fit_transform(df_pok_out_trimmed1)
# формируем DataFrame на основе массива
data_cs11_scaled =  arr_to_df(data_cs11_scaled_temp)
data_cs11_scaled


# In[78]:


data_cs11_scaled.describe()


# In[79]:


# Построение плотности распределения
def draw_kde(col_list, df1, df2, label1, label2):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    # первый график
    ax1.set_title(label1)
    sns.kdeplot(data=df1[col_list], ax=ax1)
    # второй график
    ax2.set_title(label2)
    sns.kdeplot(data=df2[col_list], ax=ax2)
    plt.show()


# In[80]:


draw_kde(['Доходы, тыс.руб.',
          'Работники, чел.',
          'Наличие тракторов, шт.',
          'Наличие комбайнов, шт.',
          'Общая площадь земли, га',
          'Урожайность зерновых, га'],
        df_pok_out_trimmed1, data_cs11_scaled, 'до масштабирования', 'после масштабирования')


# Отбор факторов

# In[81]:


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(20,10))
fig.suptitle('Корреляционная матрица:Pearson ')
sns.heatmap(data_cs11_scaled.corr(method='pearson'), cmap='YlGnBu', annot=True, fmt='.3f')


# In[82]:


data_cs11_scaled.describe ()


# In[83]:


# Проверка, есть ли отсутствующие значения
data_cs11_scaled.isnull().sum()


# In[84]:


# DataFrame не содержащий целевой признак
X = data_cs11_scaled.drop(['Доходы, тыс.руб.', 'Наличие комбайнов, шт.' , 'Урожайность зерновых, га'] , axis=1)
y = data_cs11_scaled [['Доходы, тыс.руб.']]


# In[85]:


mi = mutual_info_regression(X, y)
mi = pd.Series(mi)
#mi.index = data_cs11_scaled ['X']
mi.sort_values(ascending=False).plot.bar(figsize=(10,5))
plt.ylabel('Взаимная информация')


# In[86]:


sel_mi = SelectKBest(mutual_info_regression, k= 'all').fit( X, y)

list(zip(X, sel_mi.get_support()))


# In[87]:


# Проверка, есть ли выбросы
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(data_cs11_scaled ['Работники, чел.'], ax = axs[0])
plt2 = sns.boxplot(data_cs11_scaled ['Наличие тракторов, шт.'], ax = axs[1])
plt2 = sns.boxplot(data_cs11_scaled ['Общая площадь земли, га'], ax = axs[2])
plt.tight_layout()


# In[88]:


# Целевая переменная
sns.boxplot(data_cs11_scaled ['Доходы, тыс.руб.'])
plt.show()


# In[89]:


sns.pairplot(data_cs11_scaled, x_vars=['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га'], y_vars='Доходы, тыс.руб.')
plt.show()


# In[90]:


sns.pairplot(data_cs11_scaled.head(100), diag_kind='kde') #


# In[91]:


sns.heatmap(data_cs11_scaled.corr(), annot=True)
plt.show()


# In[92]:


X.head ()


# In[93]:


y.head ()


# In[94]:


X.shape, y.shape


# In[95]:


# Разделим выборку на обучающую и тестовую
#X_train, X_test, y_train, y_test = train_test_split(X_ALL, X_bpnup ['Модуль упругости при растяжении, ГПа'],
                                                   # test_size=0.3,
                                                   # random_state=1)
# Преобразуем массивы в DataFrame
#X_train_df = arr_to_df(X_train)
#X_test_df = arr_to_df(X_test)

#X_train_df.shape, X_test_df.shape


# In[96]:


#x_array_test=data_cs11_scaled['Выручка от реализ. зерновых на 1 га посевов, тыс.руб.'].values
#y_array_test=data_cs11_scaled['Доходы на 100 га сельхозугодий, тыс.руб.'].values
#x_array_test


# In[97]:


X1 = X.copy(deep=True)


# In[98]:


y1 = y.copy(deep=True)


# In[99]:


# Разделим выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)


# In[100]:


X_train.head()


# In[101]:


y_train.head()


# In[102]:


X_train.shape, X_test.shape


# МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ 1

# In[103]:


# Добавляем константу
X_train_full = sm.add_constant(X_train)
X_train_full


# In[104]:


# OLS - ordinary least squares
model = sm.OLS(y_train, X_train_full).fit()


# In[105]:


model.params


# In[106]:


model.summary()


# In[107]:


#plt.scatter(X_train, y_train)
#plt.plot(X_train, [0.0154 + 0.2216 * 'Работники, чел.'+ 0.1627 * 'Наличие тракторов, шт.'+ 0.5706 * 'Общая площадь земли, га'], 'r')
#plt.show()


# МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ2

# In[108]:


model = LinearRegression()


# In[109]:


model.fit(X_train, y_train)


# In[110]:


model.coef_


# In[111]:


pd.DataFrame (model.coef_, X_train.columns)


# In[112]:


y_pred = model.predict(X_test)


# In[113]:


MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)


# In[114]:


MAE


# In[115]:


MSE


# In[116]:


MAE / y_train.mean()


# СЛУЧАЙНЫЙ ЛЕС

# In[117]:


random_forest_tuning = RandomForestRegressor(random_state = 42)
param_grid = {
   'n_estimators': [100, 200, 500],
   'max_features': ['auto', 'sqrt', 'log2'],
   'max_depth' : [4,5,6],
   'criterion' :['squared_error']
}
GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5, verbose=2)
GSCV.fit(X_train, y_train)
GSCV.best_params_ 


# In[118]:


rf = GSCV.best_estimator_
rf


# In[119]:


rf = RandomForestRegressor(GSCV.best_params_)
rf


# In[120]:


rf = RandomForestRegressor(criterion='squared_error', max_depth=6, 
                           max_features='auto', n_estimators=500)


# In[121]:


rf.fit(X_train, y_train)


# In[122]:


test_predictions = rf.predict(X_test)

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[123]:


rf.score(X_test, y_test)


# In[124]:


rf.predict(X_test)


# In[125]:


np.mean((y_test - np.mean(y_test))*(y_test - np.mean(y_test)))


# In[126]:


prediction=rf.predict(X_test)


# In[135]:


np.mean ((y_test.squeeze() - prediction)*(y_test.squeeze() - prediction))


# In[133]:


pd.DataFrame(GSCV.cv_results_)


# In[134]:


plt.hist(pd.DataFrame(GSCV.cv_results_)['mean_test_score'])


# ЛАССО

# In[136]:


lassso = Lasso(random_state = 42)
param_grid = {
   'alpha': np.linspace(0, 1, 100)
}
GSCV = GridSearchCV(estimator=lassso, param_grid=param_grid, cv=10, verbose=2)
GSCV.fit(X_train, y_train)
GSCV.best_params_ 


# In[137]:


model=GSCV.best_estimator_


# In[138]:


model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[140]:


prediction=model.predict(X_test)
np.mean((y_test.squeeze() - prediction)*(y_test.squeeze() - prediction))


# МЕТОД K-БЛИЖАЙШИХ СОСЕДЕЙ

# In[141]:


knn = KNeighborsRegressor()
param_grid = {'n_neighbors': [1, 2, 5, 10, 20]}
GSCV = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, verbose=2)
GSCV.fit(X_train, y_train)
GSCV.best_params_ 


# In[142]:


knn.fit(X_train, y_train)
prediction=knn.predict(X_test)
np.mean((y_test - prediction)*(y_test - prediction))


# МОДЕЛЬ

# In[144]:


X2 = X1.copy(deep=True)


# In[145]:


y2 = y1.copy(deep=True)


# In[146]:


X2_train_val, X2_test, y2_train_val, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train_val, y2_train_val, test_size=0.2, random_state=1)


# In[147]:


X2_train.shape, X2_test.shape, X2_val.shape


# In[148]:


X2_train.head()


# In[149]:


y2_train.head()


# In[150]:


resourses = np.array(X2_train[['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га'] ])

resourses_normalizer = layers.Normalization(input_shape=[1,], axis=None)
resourses_normalizer.adapt(np.array(X2 [ ['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га'] ]))


# In[151]:


resourses


# In[152]:


resourses_model = tf.keras.Sequential([
    resourses_normalizer,
    layers.Dense(units=1)
])

resourses_model.summary()


# In[153]:


resourses [:10]


# In[160]:


resourses_normalizer2 = layers.Normalization(input_shape=[1,], axis=-1)
resourses_normalizer2.adapt(np.array(X2 [ ['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га'] ]))
resourses_model2 = tf.keras.Sequential([
    resourses_normalizer2,
    layers.Dense(units=1)
])

resourses_model2.summary()
resourses_model2.predict(resourses[:10])


# In[161]:


y2_train.values[:10] 


# In[164]:


resourses_model2.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[206]:


get_ipython().run_cell_magic('time', '', "history = resourses_model2.fit(\n    X2_train [['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га']],\n    y2_train,\n    epochs=30,\n    verbose=1,\n    validation_split = 0.2)")


# In[166]:


history.history


# In[167]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[168]:


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Эпоха')
    plt.ylabel('MAE [MPG]')
    plt.legend()
    plt.grid(True)


# In[169]:


plot_loss(history)


# In[171]:


resourses_model2.evaluate(X2_test[['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га']], y2_test, verbose=0)


# In[207]:


test_results = {}

test_results['resourses_model'] = resourses_model2.evaluate(X2_test[['Работники, чел.', 'Наличие тракторов, шт.', 'Общая площадь земли, га']], 
                                                            y2_test, verbose=0)


# In[208]:


test_results


# In[209]:


tf.linspace(0.0, 250, 251)


# In[210]:


x = tf.linspace(0.0, 250, 251)
prediction = resourses_model2.predict(x)


# Построение линейной регрессии

# In[211]:


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X2))
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])


# In[212]:


linear_model.predict(X2_train[:10])


# In[213]:


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[214]:


get_ipython().run_cell_magic('time', '', 'history = linear_model.fit(\n    X2_train,\n    y2_train,\n    epochs=30,\n    verbose=1,\n    validation_split = 0.2)')


# In[215]:


plot_loss(history)


# In[216]:


test_results['linear_model'] = linear_model.evaluate(X2_test, y2_test, verbose=0)


# In[217]:


test_results


# <b>Построение многоcлойного персептрона</b>

# In[218]:


def build_and_compile_model(norm):
    model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# In[219]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


# In[220]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    X2_train,\n    y2_train,\n    validation_split=0.2,\n    verbose=0, epochs=300)')


# In[221]:


plot_loss(history)


# In[222]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    X2_train,\n    y2_train,\n    validation_split=0.2,\n    verbose=0, epochs=300)')


# In[223]:


plot_loss(history)


# In[224]:


test_results['dnn_model'] = dnn_model.evaluate(X2_test, y2_test, verbose=0)


# In[225]:


test_results


# In[226]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# In[227]:


test_predictions = dnn_model.predict(X2_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y2_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 4]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[228]:


error = y2_test.squeeze() - test_predictions
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# In[ ]:





# In[ ]:




