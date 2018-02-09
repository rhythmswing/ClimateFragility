import numpy as np
from sklearn import linear_model
import math
import sys

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
        selectedpair = pairs[np.abs(minus) < 2 * sigma]
        treated = selectedpair[:,0]
        untreated = selectedpair[:,1]
        # print('\rleft {}/{}'.format(len(selectedpair),len(concerned_value)))

        pre_tr_ft = feats[concernedidx]
        pre_ut_ft = feats[to_matched]
        pre_indic = (np.mean(pre_tr_ft, axis=0)-np.mean(pre_ut_ft, axis=0))\
                    /(np.sqrt(0.5*(np.var(pre_tr_ft,axis=0)+np.var(pre_ut_ft,axis=0))))
        pos_tr_ft = feats[treated]
        pos_ut_ft = feats[untreated]
        pos_indic = (np.mean(pos_tr_ft, axis=0)-np.mean(pos_ut_ft, axis=0))\
                    /np.sqrt(0.5*(np.var(pos_tr_ft,axis=0)+np.var(pos_ut_ft,axis=0)))

        pos = np.sum(lb[treated])
        neg = np.sum(lb[untreated])
        print('')
        print("result:pos {}, neg {}, ratio: {}, std bias {}->{}".format(pos, neg, pos/neg,np.average(np.abs(pre_indic)),np.average(np.abs(pos_indic))))
        return [pos, neg,pre_indic,pos_indic]
