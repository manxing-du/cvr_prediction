#!/usr/bin/python
import sys
import operator
import pandas as pd

if len(sys.argv) < 5:
    print 'Usage: train.log.txt test.log.txt train.lr.txt test.lr.txt featindex.txt'
    exit(-1)


fs = ["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat", "idaffiliate","aff_type"]
    #, "idcampaign_diff_cvr_1","user_id_diff_cvr_1"]

#continous_ft = ["cvr_ratio"]

# initialize
namecol = {}
featindex = {}
maxindex = 0
fi = open(sys.argv[1], 'r')
first = True

featindex['truncate'] = maxindex
maxindex += 1

for line in fi:
    s = line.split(',')
    if first:
        first = False
        for i in range(0, len(s)):
            if s[i].strip() in fs:
                namecol[s[i].strip()] = i
                if i > 0:
                    featindex[str(i) + ':other'] = maxindex
                    maxindex += 1
            #if s[i].strip() in continous_ft:
            #    print "yes"
            #    namecol[s[i].strip()] = i
            #    maxindex += 1
        
        continue
    for f in fs:
        col = namecol[f]
        content = s[col]
        feat = str(col) + ':' + content
        if feat not in featindex:
            featindex[feat] = maxindex
            maxindex += 1



#Add one more continous feature
#print 'feature size: ' + str(maxindex + 2)

print 'feature size: ' + str(maxindex)
featvalue = sorted(featindex.iteritems(), key=operator.itemgetter(1))
fo = open(sys.argv[5], 'w')
for fv in featvalue:
    fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
fo.close()


# indexing train
print 'indexing ' + sys.argv[1]
fi = open(sys.argv[1], 'r')
fo = open(sys.argv[3], 'w')



first = True
for line in fi:
    j = 0
    if first:
        first = False
        continue
    s = line.split(',')
    fo.write(s[13].rstrip('\n')) # purchase
    index = featindex['truncate']
    fo.write(' ' + str(index) + ":1")
    for f in fs: # features
        col = namecol[f]
        content = s[col]
        feat = str(col) + ':' + content
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(' ' + str(index) + ":1")
    
    #for f in continous_ft:
    #    col=namecol[f]
    #    content = s[col]
    #    index = maxindex + j
    #    j += 1
    #    fo.write(' ' + str(index) + ":" + str(content))
    
    fo.write('\n')
fo.close()



# indexing test
print 'indexing ' + sys.argv[2]
fi = open(sys.argv[2], 'r')
fo = open(sys.argv[4], 'w')


first = True
for line in fi:
    j = 0
    if first:
        first = False
        continue
    s = line.split(',')
    fo.write(s[13].rstrip('\n') ) # click
    index = featindex['truncate']
    fo.write(' ' + str(index) + ":1")
    for f in fs:
        col = namecol[f]
        if col >= len(s):
            print 'col: ' + str(col)
            print line
        content = s[col]
        feat = str(col) + ':' + content
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(' ' + str(index) + ":1")
    
    #for f in continous_ft:
    #    col=namecol[f]
    #    content = s[col]
    #    index = maxindex + j
    #    j += 1
    #    fo.write(' ' + str(index) + ":" + str(content))

    fo.write('\n')
fo.close()

