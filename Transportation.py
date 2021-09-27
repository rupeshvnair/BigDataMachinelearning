import pandas as pd
df_1 = pd.read_csv("C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/devenv/devproj/BigDataMachinelearning/headbrain.csv")
df_1 = df_1.drop(labels="Unnamed: 0",axis = 1)

totrows = len(df_1)
totcols = len(df_1.columns)
lastcol = list(df_1.columns)
lastcol = lastcol[-1]

coltotal = {}
rowtotal = {}
for i in df_1.columns:
    coltotal[i]=df_1[i].loc[totrows-1]


coltotal.popitem()
# print(coltotal)

for i in range(len(df_1)):
    rowtotal[i]=df_1[lastcol].iloc[i]

rowtotal.popitem()
# print(rowtotal)


# coltotal = {'Delhi':2300,'Mumbai':1400}
# rowtotal = {0:1000,1:1500,2:1200}

# print(df_1)

# print(df_1)


# coltotal = {}
# rowtotal = {}
# for z in df_1.columns:
#     coltotal[z]=df_1[z].sum()
#
# for y in range(len(df_1)):
#     rowtotal[y]= df_1.loc[y].sum()
#
# print(coltotal)
# print(rowtotal)

#
a = 100000000000000


mincost =0

while (max(rowtotal.values())!=0) and (max(coltotal.values())!=0):
    for i in coltotal.keys():
        if coltotal[i]==0:
            continue
        else:
            for j in rowtotal.keys():
                if rowtotal[j]==0:
                    continue
                else:
                    if df_1[i].loc[j] <= a:
                        a = df_1[i].loc[j]
                        col = i
                        row = j

    if coltotal[col] < rowtotal[row]:
        rowtotal[row]=rowtotal[row]-coltotal[col]
        mincost= mincost + a *coltotal[col]
        coltotal[col]=0
        a = 100000000000000
    else:
        coltotal[col]=coltotal[col]-rowtotal[row]
        mincost= mincost + a *rowtotal[row]
        rowtotal[row]=0
        a = 100000000000000

print(mincost)

