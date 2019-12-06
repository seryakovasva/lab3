import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

train = pd.read_csv('./train.csv')
# x_test = pd.read_csv('./test.csv')

#x_test = train.sample(frac=0.2)
#x_train = pd.concat([train, x_test, x_test]).drop_duplicates(keep=False)

x_train , x_test = train[0:623].reset_index(drop=True), train[624:891].reset_index(drop=True)
res3 = x_test.copy()
# обработка
y_train = x_train['Survived']
y_test3 = x_test['Survived']
x_train = x_train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Survived', 'Pclass'],
                       axis=1)
x_test = x_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Survived','Pclass'],
                     axis=1)
x_train['Age'] = x_train.Age.fillna(x_train['Age'].median())
x_train['Age'] = x_train['Age'].astype(int)
x_train['Sex'] = x_train['Sex'].map({"male": 0, "female": 1})
x_test['Age'] = x_test.Age.fillna(x_test['Age'].median())
x_test['Age'] = x_test['Age'].astype(int)
x_test['Sex'] = x_test['Sex'].map({"male": 0, "female": 1})




# наивный байесовский классификатор
model = GaussianNB()
parameters = {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05],
              'priors': [[0.4, 0.6], [0.5,0.5]]}
lr = GridSearchCV(GaussianNB(), parameters, cv=5, verbose=2)
lr.fit(x_train, y_train)
# model.fit(x_train, y_train)
predicted = lr.best_estimator_.predict(x_test)
y_test1 = pd.Series(predicted)
result = lr.cv_results_
mass = result['mean_test_score']
print(mass)
print("Наилучшие значения параметров: {}".format(lr.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(lr.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(x_test, y_test1)))


result2 = lr.cv_results_
mass = str(result2['params'])
mass = mass[1:-1].split("}, ")
massRes = result2['mean_test_score']
print(mass)
print(massRes)

dataf1 = pd.DataFrame(massRes,
                      columns = ['res'])
dataf1['param'] = pd.Series(mass, index=dataf1.index)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
y = np.array(list(massRes))
# ax.plot(mass, y, linewidth = 5)
sns.countplot(x='param', data=dataf1, hue='res')
#ax.grid(True, which='major', color='k', linestyle='solid')
plt.xticks(rotation=90)
#plt.rc('xtick', labelsize=16)
#plt.rc('ytick', labelsize=20)
plt.show()

# print(res1)
# def groupAge(x):
#     r = [-1, 16, 35, 45, 65, 120]
#     g = [0, 1, 2, 3, 4]
#     x['Age'] = pd.cut(x['Age'], bins=r, labels=g)
#     return x
#
#
# def graf(x, y):
#     x.pivot_table('Age', ['Sex', 'Age'], 'Survived', 'count').plot(kind='bar', stacked=True)
#     plt.ylabel('count survived')
#     plt.xlabel('Sex, Age')
#     plt.suptitle("suptitle" + y)
#     # fig, ax = plt.subplots(figsize=(8, 6))
#     # sns.countplot(x='Survived', data=x, hue='Pclass')
#     # ax.set_ylim(0, 150)
#     # plt.title("suptitle" + y)
#     # plt.show()


res1 = x_test.copy()
res1['Survived'] = pd.Series(predicted, index=x_test.index)
res1 = res1.sort_values(by=['Age'])
#res1 = groupAge(res1)
#graf(res1, '1')

# логистическая регрессия
parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'penalty': ['l1', 'l2']}
lr = GridSearchCV(LogisticRegression(solver='liblinear', random_state= 10), parameters, cv=5, verbose=2)
lr.fit(x_train, y_train)
y_test = lr.best_estimator_.predict(x_test)
print("Наилучшие значения параметров: {}".format(lr.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(lr.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(x_test, y_test)))

res2 = x_test.copy()
res2['Survived'] = pd.Series(y_test, index=x_test.index)
res2 = res2.sort_values(by=['Age'])
#res2 = groupAge(res2)



# y = np.array(list(massRes))
# y['C'] = pd.Series(y_test3, index=x_test.index)
#
# fig = plt.figure(figsize=(13, 5))
# ax = fig.add_subplot(111)
# ax.plot(massRes, y, linewidth = 3)
# ax.grid(True, which='major', color='k', linestyle='solid')
# plt.xticks(rotation=90)
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=20)
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.countplot(x='Survived', data=x, hue='Pclass')
# ax.set_ylim(0, 150)
# plt.title("suptitle" + y)
# plt.show()

#mass['Res'] = pd.Series(massRes, index=mass.index)



#res3['Survived'] = pd.Series(y_test3, index=x_test.index)
# res3 = res3.sort_values(by=['Age'])
# res3 = groupAge(res3)
#graf(res3, "3")

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.countplot(x='Survived', data=res3, hue='Pclass')
# ax.set_ylim(0, 150)
# plt.title("Impact of Pclass on Survived")
# plt.show()
