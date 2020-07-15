# coding: utf-8
# наше всё
import numpy as np
import pandas as pd

#настройки pandas, с которыми лучше почти всегда
pd.set_option('display.max_rows', 45000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.max_colwidth', 5000)

# добавить папку с собственными библиотеками (если лни у тебя есть)
import sys
# ддя linux
sys.path.append('//home//roman//python//')
sys.path.append('//home//roman_vm//my_lib//')
# ддя windows
sys.path.append('C:\\Users\\Tsaregorodtsev.Roman\\python\\')

# собственные библиотеки
# обмен с гуглдоки
import ts_gd as gd
# получение из sql
from data_load import *

# отключим предупреждения Anaconda
# кажется, не работает
import warnings
warnings.simplefilter('ignore')

# будем отображать графики прямо в jupyter'e
# лучше не делать в линуксе
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
# лучше не делать в линуксе
# графики в svg выглядят более четкими
%config InlineBackend.figure_format = 'svg' 

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

# multistreaming
import threading

# для нормальной работы с датами
import datetime
from datetime import timedelta
from datetime import datetime
# переменная типа datetime
dtdt=datetime.strptime("2019-03-01", "%Y-%m-%d")
# изменение времени
dtdtn=dtdt+timedelta(days=15)
# привести к переменной типа str
tstr=dtdt.strftime("%Y-%m-%d")
# перевести unixtime в ообычную строку
from datetime import datetime
return datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')

# работа с файлами
# чтение текстового файла
with open('text.txt', 'r') as f:
    key = f.read()
    
# запись переменной в файл
txt='New string'
with open('readme.md', 'w') as f:
    f.write(txt)
    f.close()
    
# запись в json
import json
with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)
# json to dataframe
import json
js = json.loads(strng)
df=pd.DataFrame(js)
    

# работа с запросами
import requests
url='https://'
# если get (оно чаще)
r = requests.get(url, auth=(param1_key, param1_value))
# если post
r = requests.post(url, data = {'key':'value'})
# числовой статус, 200 - ок
r.status_code
# всё в виде текста
txt=r.text
# всё в виде json
js=r.json()

# работа с html
from bs4 import BeautifulSoup
soup = BeautifulSoup(html)
# список ссылок у нас внутри тэга div класса dib, дастаём его
lst=soup.find_all("div", {"class":"dib"})
# текст внутри тэга
lst.get_text()
# ссылки
lst.find_all("a")[0].get('href')

# работа с текстом
import re
# убираем нечисловые и небуквенные символы
txt=re.sub(r'[@#$%^&*;"(),/<>:-]', ' ', txt)
# заменить много пробелов одним
txt=re.sub(r'\s+', ' ', txt)

# разбиваем на токены
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokens=tokenizer.tokenize(txt)   

# для нормализации
# документация от дядьки-маньяка: https://pymorphy2.readthedocs.io/en/latest/user/guide.html
import pymorphy2
# говорят, тяжёлая дура
morph = pymorphy2.MorphAnalyzer()

# приводим токен к нормальной форме
morph.parse(token)[0].normal_form


# работа с pandas
# все нормальные параметры для группировки
g=df.groupby(['f1', 'f2']).agg(
                {
                    'id': lambda x: x.nunique(),
                    'revenue': np.sum
                }).reset_index().sort_values(by='id', ascending=False)

# максимум параметров для pivot
pv=df.pivot_table(index='date', columns='dd', values='revenue', aggfunc=np.sum).fillna(0).cumsum(axis=1).reset_index()

# параметр для merge, который проще скопировать, чем написать без ошибок
res=df1.merge(df2, 'left', on='device_id', suffixes=('_lft', '_rght'))

# смена формата данных для pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.float_format = '${:,.2f}'.format
df['cost'] = df['cost'].map('${:,.2f}'.format)

# правильный формат даты
df['date'] = df['date'].astype('datetime64[ns]')

# row number в pandas
df['rn']=df.sort_values(['time', 'sort_val'], ascending=False).groupby(['id']).cumcount() + 1

# дополнительный пакет для анализа
import pandas_profiling
pandas_profiling.ProfileReport(df)


# графики
# настройки pandas для приемлемого бара
df.plot(legend=False, figsize=(12,9), kind='bar', tacked=True, colormap='PuBu',edgecolor='blue',
                                        itle='Number of purchases dynamics')

# двухцветная точечная
ax = f0g1.plot.scatter(x='price', y='max_quality', color='DarkBlue', label='Had purchases', s=50)
f0g2.plot.scatter(x='price', y='max_quality', color='Red', label='No purchases', s=50,
                  ax=ax, figsize=(12,9), title='Round 0')

# точечная с переменой цвета и радиуса
import matplotlib.pyplot as plt
colormap = cm.Dark2
colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(pur['id'].drop_duplicates()))]


id_lst=pur['id'].value_counts().index

fig, ax = plt.subplots()
i=0
for id in id_lst:
    buf=pur[pur['id']==id]
    x=buf['period']
    y=buf['price']
    scale=20+buf['pur']/5
    color=colorlist[i]
    ax.scatter(x, y, c=color, s=scale, label=str(id), alpha=0.9, edgecolors='none')
    i=i+1
    
ax.legend()
ax.grid(True)

plt.show()



# машинное обучение
# строит график, считает метрики
def check(x, y1, y2, title):
    from sklearn.metrics import mean_squared_error
    print(title)
    df=pd.DataFrame([x, y1, y2]).T
    df.columns=['x', 'y1', 'y2']
    
    if len(df)>2000:
        df=df.sample(300)
    
    df=df.sort_values(by='x')
    df.set_index('x').plot(title=title)
    
    print('correlation: ', np.corrcoef(y1, y2)[0][1])
    
    print('stdev: ', mean_squared_error(y1, y2))
    
    
# OHE
color_columns=pd.get_dummies(df['Color'])

# разбиваем на выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

import xgboost
xgb = xgboost.XGBRegressor(n_estimators=ch_n_estimators, random_state=17, learning_rate=i)

# feature importances dataferame
features_df = pd.DataFrame(data=xgb.feature_importances_.reshape(1, -1), columns=X_train.columns).sort_values(axis=1, by=[0], ascending=False).T
features_df=features_df.reset_index()
features_df.columns=['feature', 'importance']



# не питон
# командная строка

# конвертировать ноутбук в обычный питон
jupyter nbconvert --to python cron_experiments.ipynb

# работа с git
git status    
git checkout master
git pull
git add filename.ext
# git add -A
git commit -m "Your comment"
git branch fix/AN-14-roma_good_bye
git checkout fix/AN-14-roma_good_bye
git push origin fix/AN-14-roma_good_bye
# пройти по ссылке и сделать реквест
