
# coding: utf-8

# In[140]:

import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


# In[147]:

def read_fsi():
    csvFile = open("modified_fsi-2017.csv", "r")
    reader = csv.reader(csvFile)
    attr_dic = {}
    score_dic = {}
    attributes = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            attributes = item[4:]
            continue
        attr_dic[item[0]] = [float(i) for i in item[4:]]
        score_dic[item[0]] = float(item[3])
    csvFile.close()
    return attr_dic, score_dic, attributes


# In[148]:

def input_fsi():
    attr_dic, score_dic, attributes = read_fsi()
    for (idx, item) in enumerate(attr_dic):
        if idx == 0:
            attr = np.array(attr_dic[item])
        new_attr = np.array(attr_dic[item])
        attr = np.vstack((attr, new_attr))
    for (idx, item) in enumerate(score_dic):
        if idx == 0:
            score = np.array(score_dic[item])
        new_score = np.array(score_dic[item])
        score = np.vstack((score, new_score))
    return attr, score


# In[153]:

def read_epi():
    csvFile = open("modified_EPI.csv", "r")
    reader = csv.reader(csvFile)
    attr_dic = {}
    score_dic = {}
    attributes = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            attributes = item[3:10]
            continue
        attr_dic[item[2]] = [float(i) for i in item[3:10]]
        score_dic[item[2]] = float(item[10])
    csvFile.close()
    return attr_dic, score_dic, attributes


# In[154]:

def input_epi():
    attr_dic, score_dic, attributes = read_epi()
    for (idx, item) in enumerate(attr_dic):
        if idx == 0:
            attr = np.array(attr_dic[item])
        new_attr = np.array(attr_dic[item])
        attr = np.vstack((attr, new_attr))
    for (idx, item) in enumerate(score_dic):
        if idx == 0:
            score = np.array(score_dic[item])
        new_score = np.array(score_dic[item])
        score = np.vstack((score, new_score))
    return attr, score


# In[1126]:

def share_country():
    '''
    Find shared countries of FSI and EPI data.
    Return:
        FSI_data -- FSI data sorted according to countries.
        EPI_data -- EPI data sorted according to countries.
    Note that these two dictionaries have the same number of countries.
    And they also have the aligned attributes based on countries order.
    '''
    fsi_attrdic, fsi_scoredic, fsi_attributes = read_fsi()
    epi_attrdic, epi_scoredic, epi_attributes = read_epi()
    share_key = []
    cnt = 0
    # First, delete all keys and values in fsi but not in epi
    for (idx, item) in enumerate(fsi_attrdic):
        epi_key = list(epi_attrdic.keys())
        if item in epi_key:
            if cnt == 0:
                fsi_attr = np.array(fsi_attrdic[item]).reshape(1,-1)
                fsi_score = np.array(fsi_scoredic[item])
                share_key.append(item)
            else:
                share_key.append(item)
                fsi_attr = np.vstack((fsi_attr, fsi_attrdic[item]))
                fsi_score = np.vstack((fsi_score, fsi_scoredic[item]))
            cnt += 1
    cnt = 0
    # Second, delete all keys and values in epi but not in fsi
    for (idx, item) in enumerate(epi_attrdic):
        fsi_key = list(fsi_attrdic.keys())
        if item in fsi_key:
            if cnt == 0:
                epi_attr = np.array(epi_attrdic[item]).reshape(1,-1)
                epi_score = np.array(epi_scoredic[item])
            else:
                epi_attr = np.vstack((epi_attr, epi_attrdic[item]))
                epi_score = np.vstack((epi_score, epi_scoredic[item]))
            cnt += 1
    df = pd.DataFrame(data = {'country': share_key,
                              'FSI': fsi_score.ravel(),
                              'EPI': epi_score.ravel(),
                              fsi_attributes[0]: fsi_attr[:,0].ravel(),
                              fsi_attributes[1]: fsi_attr[:,1].ravel(), 
                              fsi_attributes[2]: fsi_attr[:,2].ravel(),
                              fsi_attributes[3]: fsi_attr[:,3].ravel(), 
                              fsi_attributes[4]: fsi_attr[:,4].ravel(),
                              fsi_attributes[5]: fsi_attr[:,5].ravel(), 
                              fsi_attributes[6]: fsi_attr[:,6].ravel(),
                              fsi_attributes[7]: fsi_attr[:,7].ravel(), 
                              fsi_attributes[8]: fsi_attr[:,8].ravel(),
                              fsi_attributes[9]: fsi_attr[:,9].ravel(),
                              fsi_attributes[10]: fsi_attr[:,10].ravel(), 
                              fsi_attributes[11]: fsi_attr[:,11].ravel(),
                              epi_attributes[0]: epi_attr[:,0].ravel(),
                              epi_attributes[1]: epi_attr[:,1].ravel(), 
                              epi_attributes[2]: epi_attr[:,2].ravel(),
                              epi_attributes[3]: epi_attr[:,3].ravel(), 
                              epi_attributes[4]: epi_attr[:,4].ravel(),
                              epi_attributes[5]: epi_attr[:,5].ravel(), 
                              epi_attributes[6]: epi_attr[:,6].ravel()})
    df.to_csv('shared_data.csv')
    fsi_df = pd.DataFrame(data = {'country': share_key,
                                  'FSI': fsi_score.ravel(),
                                  fsi_attributes[0]: fsi_attr[:,0].ravel(),
                                  fsi_attributes[1]: fsi_attr[:,1].ravel(), 
                                  fsi_attributes[2]: fsi_attr[:,2].ravel(),
                                  fsi_attributes[3]: fsi_attr[:,3].ravel(), 
                                  fsi_attributes[4]: fsi_attr[:,4].ravel(),
                                  fsi_attributes[5]: fsi_attr[:,5].ravel(), 
                                  fsi_attributes[6]: fsi_attr[:,6].ravel(),
                                  fsi_attributes[7]: fsi_attr[:,7].ravel(), 
                                  fsi_attributes[8]: fsi_attr[:,8].ravel(),
                                  fsi_attributes[9]: fsi_attr[:,9].ravel(),
                                  fsi_attributes[10]: fsi_attr[:,10].ravel(), 
                                  fsi_attributes[11]: fsi_attr[:,11].ravel()
                                  })
    fsi_df.to_csv('shared_fsi.csv')
    epi_df = pd.DataFrame(data = {'country': share_key,
                                  'EPI': epi_score.ravel(),
                                  epi_attributes[0]: epi_attr[:,0].ravel(),
                                  epi_attributes[1]: epi_attr[:,1].ravel(), 
                                  epi_attributes[2]: epi_attr[:,2].ravel(),
                                  epi_attributes[3]: epi_attr[:,3].ravel(), 
                                  epi_attributes[4]: epi_attr[:,4].ravel(),
                                  epi_attributes[5]: epi_attr[:,5].ravel(), 
                                  epi_attributes[6]: epi_attr[:,6].ravel()
                                  })
    epi_df.to_csv('shared_epi.csv')
    return share_key, fsi_attr, epi_attr


# In[1134]:

country, fsi_attr, epi_attr = share_country()


# In[1135]:

def input_result():
    csvFile = open("shared_data.csv", "r")
    reader = csv.reader(csvFile)
    fsi_dic = {}
    epi_dic = {}
    fsi = []
    epi = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            continue
        fsi_dic[item[1]] = float(item[2])
        epi_dic[item[1]] = float(item[3])
        fsi.append(float(item[2]))
        epi.append(float(item[3]))
    csvFile.close()
    return fsi, epi


# In[1137]:

fsi, epi = input_result()


# In[1138]:

plt.scatter(fsi, epi)
plt.show()


# In[604]:

from sklearn.cluster import KMeans


# In[605]:

def cluster(data, n_clusters):
    '''
    Cluster data into n_clusters categories.
    Input:
        data -- The input data to be clustered.
        n_clusters -- The number of categories to be clustered.
    Return:
        label_pred -- The category array corresponding to every data.
    '''
    estimator = KMeans(n_clusters = n_clusters)
    estimator.fit(data)
    label_pred = estimator.labels_
    print(label_pred)
    return label_pred


# In[1139]:

FSI_data = np.array([fsi]).reshape([-1,1])
EPI_data = np.array([epi]).reshape([-1,1])
#FSI_label = cluster(FSI_data, 2)
#EPI_label = cluster(EPI_data, 2)
FSI_label = (FSI_data>80).ravel()
EPI_label = (EPI_data>50).ravel()
plt.scatter(FSI_data[FSI_label == 1], EPI_data[FSI_label == 1], c = 'r')
plt.scatter(FSI_data[FSI_label == 0], EPI_data[FSI_label == 0], c = 'b')
plt.show()


# In[1140]:

plt.scatter(FSI_data[EPI_label == 1], EPI_data[EPI_label == 1], c = 'r')
plt.scatter(FSI_data[EPI_label == 0], EPI_data[EPI_label == 0], c = 'b')
plt.show()


# In[1141]:

def find_bound(data, label, mode = 1):
    '''
    Find the boundary of clusters.
    Input:
        data -- The input data to be clustered.
        label -- The label corresponding to the input data.
        mode -- mode = 1 means we need to find the minimum bound of data with label = 1 while maximum bound for label = 0.
                mode = 0 means we need to find the maximum bound of data with label = 0 while maximum bound for label = 1.
    Return:
        bound -- The boundary of 1-dim cluster.
    '''
    if mode:
        label1_min = min(data[label == 1])
        label0_max = max(data[label == 0])
        bound = (label1_min+label0_max)/2
    else:
        label1_max = max(data[label == 1])
        label0_min = min(data[label == 0])
        bound = (label1_max+label0_min)/2
    return bound


# In[1142]:

FSI_bound = find_bound(FSI_data, FSI_label)
EPI_bound = find_bound(EPI_data, EPI_label)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 1)], EPI_data[(FSI_label == 1) & (EPI_label == 1)], c = 'r')
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 0)], EPI_data[(FSI_label == 1) & (EPI_label == 0)], c = 'b')
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 1)], EPI_data[(FSI_label == 0) & (EPI_label == 1)], c = 'm')
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 0)], EPI_data[(FSI_label == 0) & (EPI_label == 0)], c = 'y')
FSI_bound_y = np.linspace(min(EPI_data), max(EPI_data), 1000)
FSI_bound_x = np.full(FSI_bound_y.shape, FSI_bound)
EPI_bound_x = np.linspace(min(FSI_data), max(FSI_data), 1000)
EPI_bound_y = np.full(EPI_bound_x.shape, EPI_bound)
plt.plot(FSI_bound_x, FSI_bound_y, c = 'k')
plt.plot(EPI_bound_x, EPI_bound_y, c = 'k')
plt.xlabel('FSI')
plt.ylabel('EPI')
plt.show()


# In[725]:

from mlxtend.plotting import plot_decision_regions


# In[774]:

def logistic_util(X, Y, test_size, random_state):
    '''
    Implement logistic regression with sklearn.
    Input:
        X -- The independent variables set.
        Y -- The dependent variable/target set.
        test_size -- Test size of the whole set.
        random_state -- Param for train_test_split.
    Output:
        lr -- Logistic Regression Result.
    '''
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    #Y_train, Y_test = Y_train.ravel(), Y_test.ravel()
    #X_train, Y_train = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    #Y_train = Y_train.ravel(), Y_test.ravel()
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    X_train = X
    Y_train = Y
    # Normalize the data
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = np.array(list(sc.transform(X_train)))
    #X_test_std = np.array(list(sc.transform(X_test)))
    # Combine the train and test data
    #X_combined_std = np.vstack((X_train_std, X_test_std))
    #Y_combined = np.hstack((Y_train, Y_test))
    # Implement logistic regression
    lr = LogisticRegression(C = 0.01, random_state = 0)
    lr.fit(X_train_std, Y_train)
    #print(lr.predict_proba(X_test_std[2,:].reshape(1,-1))[0,0], Y_test[2])
    scores = cross_val_score(lr, X, Y, cv = 5, scoring = 'accuracy')
    print(scores)
    return lr, X_train_std, Y_train
    '''
    plot_decision_regions(X_combined_std, Y_combined, clf = lr, res = 0.02)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc = 'upper left')
    plt.show()
    '''


# In[775]:

from sklearn.cross_validation import cross_val_score


# In[1167]:

print(fsi_attr[0])


# In[1143]:

lr1, fsi_attr_std, _ = logistic_util(fsi_attr, EPI_label, 0, 0)
probE0_H = lr1.predict_proba(fsi_attr_std)[:,0]
probE1_H = np.ones(probE0_H.shape)-probE0_H


# In[1144]:

lr2, fsi_attr_std, _ = logistic_util(fsi_attr, (EPI_label == 0) & (FSI_label == 1), 0, 0)
prob_other = lr2.predict_proba(fsi_attr_std)[:,0]
probE0F1_H = np.ones(prob_other.shape)-prob_other


# In[1145]:

lr3, fsi_attr_std, _ = logistic_util(fsi_attr, (EPI_label == 1) & (FSI_label == 1), 0, 0)
prob_other = lr3.predict_proba(fsi_attr_std)[:,0]
probE1F1_H = np.ones(prob_other.shape)-prob_other


# In[1146]:

probF1_E0H = probE0F1_H / probE0_H
probF1_E1H = probE1F1_H / probE1_H
print(probE0F1_H[EPI_label == 0], probE0_H[EPI_label == 0])


# In[1147]:

fragile_value = []
for i in range(len(fsi_attr_std)):
    if EPI_label[i] == 1:
        fragile_value.append(probF1_E1H[i])
    else:
        fragile_value.append(probF1_E0H[i])


# In[1148]:

fragile = np.array(fragile_value)
country = np.array(country)
data_num = len(fragile_value)
df_fragile = pd.DataFrame(data = {'country': country.reshape(data_num),
                                  'fragile': fragile.reshape(data_num),
                                  'FSI': FSI_data.reshape(data_num),
                                  'EPI': EPI_data.reshape(data_num)})


# In[1149]:

df_fragile.sort_values('fragile',axis = 0,ascending = 'False')
df_fragile.to_csv('fragile.csv')


# In[736]:

fragile_value = np.array(fragile_value).reshape(-1,1)
sc = StandardScaler()
sc.fit(fragile_value)
fragile_std = (255*(np.array(list(sc.transform(fragile_value)))+1)/2).astype(np.uint8)
fragile_std = fragile_std.reshape(len(fragile_std))
print(fragile_std)


# In[578]:

fragile_value = np.array(fragile_value)
#fragile_value = np.tanh(fragile_value)
stretch = int(255/max(fragile_value))
fragile_std = (fragile_value*stretch).astype(np.uint8)
fragile_std = fragile_std.reshape(len(fragile_std))
print(fragile_std)


# In[1150]:

fragile = np.array(fragile_value)
sort_index = fragile.argsort()


# In[1151]:

import seaborn as sn
pal = sn.color_palette("GnBu_d", 149)


# In[1152]:

color = []
for i in range(len(fragile_value)):
    color.append(pal[sort_index[i]])


# In[1153]:

FSI_bound = find_bound(FSI_data, FSI_label)
EPI_bound = find_bound(EPI_data, EPI_label)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 1)], EPI_data[(FSI_label == 1) & (EPI_label == 1)], c = color)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 0)], EPI_data[(FSI_label == 1) & (EPI_label == 0)], c = color)
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 1)], EPI_data[(FSI_label == 0) & (EPI_label == 1)], c = color)
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 0)], EPI_data[(FSI_label == 0) & (EPI_label == 0)], c = color)
FSI_bound_y = np.linspace(min(EPI_data), max(EPI_data), 1000)
FSI_bound_x = np.full(FSI_bound_y.shape, FSI_bound)
EPI_bound_x = np.linspace(min(FSI_data), max(FSI_data), 1000)
EPI_bound_y = np.full(EPI_bound_x.shape, EPI_bound)
plt.plot(FSI_bound_x, FSI_bound_y, c = 'k')
plt.plot(EPI_bound_x, EPI_bound_y, c = 'k')
plt.xlabel('FSI')
plt.ylabel('EPI')
plt.show()


# In[404]:

print(probE1_H[129])
print(probE1F1_H[129])
print(probF1_E1H[129], fragile_value[129])


# In[422]:

print(probE1_H[10], EPI_label[10])


# In[1154]:

def reverse_num(FSI_data, EPI_data, fragile):
    cnt = 0
    rate = []
    for i in range(len(FSI_data)):
        mask = (EPI_data > EPI_data[i]+5) & (FSI_data+1 < FSI_data[i]) & (fragile > fragile[i])
        rate.append(1.0*sum(mask)/(len(FSI_data)-1))
    avg_rate = np.mean(rate)
    print(avg_rate)


# In[1155]:

n_samples = len(FSI_data)
reverse_num(FSI_data.reshape(n_samples), EPI_data.reshape(n_samples), fragile.reshape(n_samples))


# In[844]:

from sklearn import linear_model
import math
import sys


# In[1119]:

#  implementation of propensity score matching algorithm
#  input:
#   feats: m * n array
#   label: m * 1 binary array, 1 -> treated label, 0 -> untreated label
#   lb : m * 1 binary array, labels of samples, 1 -> positive sample, 0 -> negative sample
#   trim : leave the samples which have the probability in [trim, 1-trim]
#  output:
#   pos : number of positive samples after matching
#   neg : number of negative samples after matching
#   pre_indic : m * 1 array, std bias of treatment and untreatment before matching,
#       each dimension is correpondent to a feature.
#   post_indic : m * 1 array, std bias of each feature after matching

def propensity_score_matching(feats,label, lb, trim=0.01):
        concernedidx = np.array(range(len(lb)))[label == 1]
        to_matched = np.array(range(len(lb)))[label != 1]

        clf = linear_model.LogisticRegression(solver='sag', max_iter=5000)
        #print(clf.fit(feats, label))
        clf.fit(feats,label)
        predict_proba = clf.predict_proba(feats)[:,1]

        overlap_range_min = trim
        overlap_range_max = 1-trim
        # print overlap_range_min,overlap_range_max,'\r'
        concerned_value = {s:predict_proba[s] for s in concernedidx if overlap_range_min<=predict_proba[s]<=overlap_range_max}
        match_value = {s:predict_proba[s] for s in to_matched if overlap_range_min<=predict_proba[s]<=overlap_range_max}
        concerned_value = list(sorted(concerned_value.items(), key=lambda x:x[1]))
        match_value = list(sorted(match_value.items(), key=lambda x:x[1]))

        pairs = []
        curpos = 0
        for k, value in enumerate(concerned_value):
            if k % 1000 == 0:
                sys.stdout.write('\r{}/{} nodes processed'.format(k,len(concerned_value)))
                sys.stdout.flush()
            idx, prob = value
            while curpos < len(match_value) and prob > match_value[curpos][1]:
                curpos += 1
            if curpos == 0:
                tmp = 0
            elif curpos == len(match_value):
                tmp = curpos - 1
            else:
                tmp = curpos -1 if math.fabs(match_value[curpos-1][1] - prob) < math.fabs(match_value[curpos][1] - prob) else curpos
            pairs.append([idx, match_value[tmp][0]])
        pairs = np.array(pairs)
        minus = np.array([predict_proba[a] - predict_proba[b] for a,b in pairs])
        sigma = np.std(minus)
        selectedpair = pairs[np.abs(minus) < 2*sigma]
        treated = selectedpair[:,0]
        untreated = selectedpair[:,1]
        #print('\rleft {}/{}'.format(len(selectedpair),len(concerned_value)))

        pre_tr_ft = feats[concernedidx]
        pre_ut_ft = feats[to_matched]
        pre_indic = (np.mean(pre_tr_ft, axis=0)-np.mean(pre_ut_ft, axis=0))                    /(np.sqrt(0.5*(np.var(pre_tr_ft,axis=0)+np.var(pre_ut_ft,axis=0))))
        pos_tr_ft = feats[treated]
        pos_ut_ft = feats[untreated]
        pos_indic = (np.mean(pos_tr_ft, axis=0)-np.mean(pos_ut_ft, axis=0))                    /np.sqrt(0.5*(np.var(pos_tr_ft,axis=0)+np.var(pos_ut_ft,axis=0)))

        pos = np.sum(lb[treated])
        neg = np.sum(lb[untreated])
        print('')
        print("result:pos {}, neg {}, ratio: {}, std bias {}->{}".format(pos, neg, pos/neg,np.average(np.abs(pre_indic)),np.average(np.abs(pos_indic))))
        return [treated, untreated, pre_indic, pos_indic, predict_proba]


# In[1156]:

treated, untreated, pre_indic, pos_indic, pro_score = propensity_score_matching(fsi_attr, EPI_label.flatten(), FSI_label.flatten())
print(np.mean(fragile[untreated]-fragile[treated]))
diff = fragile[untreated]-fragile[treated]


# In[1157]:

plt.hist(diff)
np.std(diff)
plt.show()


# In[1158]:

plt.hist(pro_score[treated],alpha=0.7,bins=8)
plt.hist(pro_score[untreated],alpha=0.7,bins=8)
plt.show()


# In[959]:

F_E1 = np.mean(fragile[EPI_label == 1])
F_E0 = np.mean(fragile[EPI_label == 0])
F_diff = F_E0-F_E1
print(F_diff)


# In[960]:

def find_comp(treated, untreated, FSI_data, EPI_data, fragile):
    df = pd.DataFrame(data = {'country1': list(df_fragile['country'][treated]),
                              'country2': list(df_fragile['country'][untreated]),
                              'fragility1': fragile[treated].flatten(),
                              'fragility2': fragile[untreated].flatten()})
    df.to_csv('comparison.csv')


# In[1159]:

find_comp(treated, untreated, FSI_data, EPI_data, fragile)


# In[1160]:

FSI_label = FSI_label.astype(np.uint8)
NEPI_label = (np.ones(EPI_label.shape)-EPI_label).astype(np.uint8)
label = NEPI_label | FSI_label
treated, untreated, pre_indic, pos_indic, pro_score = propensity_score_matching(fsi_attr, label.flatten(), FSI_label.flatten())
print(np.mean(fragile[treated]-fragile[untreated]))
diff = fragile[treated]-fragile[untreated]


# In[1161]:

plt.hist(diff)
np.std(diff)
plt.show()


# In[1162]:

plt.hist(pro_score[treated],alpha=0.7,bins=5)
plt.hist(pro_score[untreated],alpha=0.7,bins=5)
plt.show()


# In[1010]:

def find_comp(treated, untreated, FSI_data, EPI_data, fragile):
    df = pd.DataFrame(data = {'country1': list(df_fragile['country'][treated]),
                              'country2': list(df_fragile['country'][untreated]),
                              'FSI1': FSI_data[treated].flatten(),
                              'FSI2': FSI_data[untreated].flatten(),
                              'EPI1': EPI_data[treated].flatten(),
                              'EPI2': EPI_data[untreated].flatten(),
                              'fragility1': fragile[treated].flatten(),
                              'fragility2': fragile[untreated].flatten(),
                              'pro_score': pro_score[treated]-pro_score[untreated]})
    df.to_csv('comparison.csv')


# In[1163]:

find_comp(treated, untreated, FSI_data, EPI_data, fragile)


# In[1014]:

def read_allfsi():
    csvFile = open("all_fsi.csv", "r")
    reader = csv.reader(csvFile)
    attr_dic = {}
    score_dic = {}
    attributes = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            attributes = item[4:]
            continue
        attr_dic[item[0]+item[1]] = [float(i) for i in item[4:]]
        score_dic[item[0]+item[1]] = float(item[3])
    csvFile.close()
    return attr_dic, score_dic, attributes


# In[1021]:

def read_allepi():
    csvFile = open("all_epi.csv", "r")
    reader = csv.reader(csvFile)
    score_dic = {}
    attributes = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            attributes = [int(i) for i in item[2:]]
            continue
        for i in range(len(attributes)):
            score_dic[item[1]+str(attributes[i])] = float(item[2+i])
    csvFile.close()
    return score_dic, attributes


# In[1026]:

fsi_attrdic, fsi_scoredic, fsi_attributes = read_allfsi()


# In[1027]:

epi_scoredic, epi_attributes = read_allepi()


# In[1038]:

def share_country_backdata():
    '''
    Find shared countries of all the FSI and EPI data in the past years.
    Return:
        FSI_data -- FSI data sorted according to countries.
        EPI_data -- EPI data sorted according to countries.
    Note that these two dictionaries have the same number of countries.
    And they also have the aligned attributes based on countries order.
    '''  
    fsi_attrdic, fsi_scoredic, fsi_attributes = read_allfsi()
    epi_scoredic, epi_attributes = read_allepi()
    share_key = []
    cnt = 0
    # First, delete all keys and values in fsi but not in epi
    for (idx, item) in enumerate(fsi_attrdic):
        epi_key = list(epi_scoredic.keys())
        if item in epi_key:
            if cnt == 0:
                fsi_attr = np.array(fsi_attrdic[item]).reshape(1,-1)
                fsi_score = np.array(fsi_scoredic[item])
                epi_score = np.array(epi_scoredic[item])
                share_key.append(item)
            else:
                share_key.append(item)
                fsi_attr = np.vstack((fsi_attr, fsi_attrdic[item]))
                fsi_score = np.vstack((fsi_score, fsi_scoredic[item]))
                epi_score = np.vstack((epi_score, epi_scoredic[item]))
            cnt += 1
    df = pd.DataFrame(data = {'country': share_key,
                              'FSI': fsi_score.ravel(),
                              'EPI': epi_score.ravel(),
                              fsi_attributes[0]: fsi_attr[:,0].ravel(),
                              fsi_attributes[1]: fsi_attr[:,1].ravel(), 
                              fsi_attributes[2]: fsi_attr[:,2].ravel(),
                              fsi_attributes[3]: fsi_attr[:,3].ravel(), 
                              fsi_attributes[4]: fsi_attr[:,4].ravel(),
                              fsi_attributes[5]: fsi_attr[:,5].ravel(), 
                              fsi_attributes[6]: fsi_attr[:,6].ravel(),
                              fsi_attributes[7]: fsi_attr[:,7].ravel(), 
                              fsi_attributes[8]: fsi_attr[:,8].ravel(),
                              fsi_attributes[9]: fsi_attr[:,9].ravel(),
                              fsi_attributes[10]: fsi_attr[:,10].ravel(), 
                              fsi_attributes[11]: fsi_attr[:,11].ravel()\
                             })
    df.to_csv('shared_alldata.csv')
    fsi_df = pd.DataFrame(data = {'country': share_key,
                                  'FSI': fsi_score.ravel(),
                                  fsi_attributes[0]: fsi_attr[:,0].ravel(),
                                  fsi_attributes[1]: fsi_attr[:,1].ravel(), 
                                  fsi_attributes[2]: fsi_attr[:,2].ravel(),
                                  fsi_attributes[3]: fsi_attr[:,3].ravel(), 
                                  fsi_attributes[4]: fsi_attr[:,4].ravel(),
                                  fsi_attributes[5]: fsi_attr[:,5].ravel(), 
                                  fsi_attributes[6]: fsi_attr[:,6].ravel(),
                                  fsi_attributes[7]: fsi_attr[:,7].ravel(), 
                                  fsi_attributes[8]: fsi_attr[:,8].ravel(),
                                  fsi_attributes[9]: fsi_attr[:,9].ravel(),
                                  fsi_attributes[10]: fsi_attr[:,10].ravel(), 
                                  fsi_attributes[11]: fsi_attr[:,11].ravel()
                                  })
    fsi_df.to_csv('shared_allfsi.csv')
    epi_df = pd.DataFrame(data = {'country': share_key,
                                  'EPI': epi_score.ravel()
                                  })
    epi_df.to_csv('shared_allepi.csv')
    return share_key, fsi_attr


# In[1040]:

share_key, fsi_attr = share_country_backdata()


# In[1041]:

def input_allresult():
    csvFile = open("shared_alldata.csv", "r")
    reader = csv.reader(csvFile)
    fsi_dic = {}
    epi_dic = {}
    fsi = []
    epi = []
    for item in reader:
        # ignore the first line
        if reader.line_num == 1:
            continue
        fsi_dic[item[1]] = float(item[2])
        epi_dic[item[1]] = float(item[3])
        fsi.append(float(item[2]))
        epi.append(float(item[3]))
    csvFile.close()
    return fsi, epi


# In[1042]:

fsi, epi = input_allresult()


# In[1096]:

FSI_data = np.array([fsi]).reshape([-1,1])
EPI_data = np.array([epi]).reshape([-1,1])
#FSI_label = cluster(FSI_data, 2)
#EPI_label = cluster(EPI_data, 2)
FSI_label = (FSI_data>84).ravel()
EPI_label = (EPI_data>60).ravel()
plt.scatter(FSI_data[FSI_label == 1], EPI_data[FSI_label == 1], c = 'r')
plt.scatter(FSI_data[FSI_label == 0], EPI_data[FSI_label == 0], c = 'b')
plt.show()


# In[1097]:

plt.scatter(FSI_data[EPI_label == 1], EPI_data[EPI_label == 1], c = 'r')
plt.scatter(FSI_data[EPI_label == 0], EPI_data[EPI_label == 0], c = 'b')
plt.show()


# In[1098]:

FSI_bound = find_bound(FSI_data, FSI_label)
EPI_bound = find_bound(EPI_data, EPI_label)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 1)], EPI_data[(FSI_label == 1) & (EPI_label == 1)], c = 'r')
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 0)], EPI_data[(FSI_label == 1) & (EPI_label == 0)], c = 'b')
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 1)], EPI_data[(FSI_label == 0) & (EPI_label == 1)], c = 'm')
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 0)], EPI_data[(FSI_label == 0) & (EPI_label == 0)], c = 'y')
FSI_bound_y = np.linspace(min(EPI_data), max(EPI_data), 1000)
FSI_bound_x = np.full(FSI_bound_y.shape, FSI_bound)
EPI_bound_x = np.linspace(min(FSI_data), max(FSI_data), 1000)
EPI_bound_y = np.full(EPI_bound_x.shape, EPI_bound)
plt.plot(FSI_bound_x, FSI_bound_y, c = 'k')
plt.plot(EPI_bound_x, EPI_bound_y, c = 'k')
plt.xlabel('FSI')
plt.ylabel('EPI')
plt.show()


# In[1099]:

print(len(FSI_label[FSI_label==1])/len(FSI_label), len(EPI_label[EPI_label == 1])/len(EPI_label))


# In[1100]:

lr1, fsi_attr_std, _ = logistic_util(fsi_attr, EPI_label, 0, 0)
probE0_H = lr1.predict_proba(fsi_attr_std)[:,0]
probE1_H = np.ones(probE0_H.shape)-probE0_H
lr2, fsi_attr_std, _ = logistic_util(fsi_attr, (EPI_label == 0) & (FSI_label == 1), 0, 0)
prob_other = lr2.predict_proba(fsi_attr_std)[:,0]
probE0F1_H = np.ones(prob_other.shape)-prob_other
lr3, fsi_attr_std, _ = logistic_util(fsi_attr, (EPI_label == 1) & (FSI_label == 1), 0, 0)
prob_other = lr3.predict_proba(fsi_attr_std)[:,0]
probE1F1_H = np.ones(prob_other.shape)-prob_other
probF1_E0H = probE0F1_H / probE0_H
probF1_E1H = probE1F1_H / probE1_H
print(probE0F1_H[EPI_label == 0], probE0_H[EPI_label == 0])


# In[1101]:

fragile_value = []
for i in range(len(fsi_attr_std)):
    if EPI_label[i] == 1:
        fragile_value.append(probF1_E1H[i])
    else:
        fragile_value.append(probF1_E0H[i])
fragile = np.array(fragile_value)
share_key = np.array(share_key)
data_num = len(fragile_value)
df_fragile = pd.DataFrame(data = {'country': share_key.reshape(data_num),
                                  'fragile': fragile.reshape(data_num),
                                  'FSI': FSI_data.reshape(data_num),
                                  'EPI': EPI_data.reshape(data_num)})


# In[1102]:

df_fragile.sort_values('fragile',axis = 0,ascending = 'False')
df_fragile.to_csv('fragile_all.csv')


# In[1103]:

print(data_num)


# In[1104]:

import seaborn as sn
pal = sn.color_palette("GnBu_d", data_num)
fragile = np.array(fragile_value)
sort_index = fragile.argsort()
color = []
for i in range(len(fragile_value)):
    color.append(pal[sort_index[i]])


# In[1105]:

FSI_bound = find_bound(FSI_data, FSI_label)
EPI_bound = find_bound(EPI_data, EPI_label)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 1)], EPI_data[(FSI_label == 1) & (EPI_label == 1)], c = color)
plt.scatter(FSI_data[(FSI_label == 1) & (EPI_label == 0)], EPI_data[(FSI_label == 1) & (EPI_label == 0)], c = color)
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 1)], EPI_data[(FSI_label == 0) & (EPI_label == 1)], c = color)
plt.scatter(FSI_data[(FSI_label == 0) & (EPI_label == 0)], EPI_data[(FSI_label == 0) & (EPI_label == 0)], c = color)
FSI_bound_y = np.linspace(min(EPI_data), max(EPI_data), 1000)
FSI_bound_x = np.full(FSI_bound_y.shape, FSI_bound)
EPI_bound_x = np.linspace(min(FSI_data), max(FSI_data), 1000)
EPI_bound_y = np.full(EPI_bound_x.shape, EPI_bound)
plt.plot(FSI_bound_x, FSI_bound_y, c = 'k')
plt.plot(EPI_bound_x, EPI_bound_y, c = 'k')
plt.xlabel('FSI')
plt.ylabel('EPI')
plt.show()


# In[1106]:

n_samples = len(FSI_data)
reverse_num(FSI_data.reshape(n_samples), EPI_data.reshape(n_samples), fragile.reshape(n_samples))


# In[1120]:

FSI_label = FSI_label.astype(np.uint8)
NEPI_label = (np.ones(EPI_label.shape)-EPI_label).astype(np.uint8)
label = NEPI_label | FSI_label
treated, untreated, pre_indic, pos_indic, pro_score = propensity_score_matching(fsi_attr,label.flatten(), FSI_label.flatten())
print(np.mean(fragile[treated]-fragile[untreated]))
diff = fragile[treated]-fragile[untreated]


# In[1121]:

plt.hist(diff)
np.std(diff)
plt.show()


# In[1123]:

plt.hist(pro_score[treated],alpha=0.7,bins=10)
plt.hist(pro_score[untreated],alpha=0.7,bins=10)
plt.show()


# In[1124]:

def find_comp(treated, untreated, FSI_data, EPI_data, fragile):
    df = pd.DataFrame(data = {'country1': list(df_fragile['country'][treated]),
                              'country2': list(df_fragile['country'][untreated]),
                              'FSI1': FSI_data[treated].flatten(),
                              'FSI2': FSI_data[untreated].flatten(),
                              'EPI1': EPI_data[treated].flatten(),
                              'EPI2': EPI_data[untreated].flatten(),
                              'fragility1': fragile[treated].flatten(),
                              'fragility2': fragile[untreated].flatten(),
                              'pro_score': pro_score[treated]-pro_score[untreated]})
    df.to_csv('comparison_all.csv')


# In[1125]:

find_comp(treated, untreated, FSI_data, EPI_data, fragile)


# In[ ]:



