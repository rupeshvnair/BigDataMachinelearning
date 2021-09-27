import pandas as pd
#the input is assumed to be through a csv file
# so the below command is used to store it to a dataframe
df_1 = pd.read_csv("C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/MSL/Python code/input.csv")

#the column with names of rows is removed as it is not required
df_1 = df_1.drop(labels="Unnamed: 0",axis = 1)


#the total rows and total columns of the dataset is measured
totrows = len(df_1)
totcols = len(df_1.columns)

#the supply column values are stored to dataframe
lastcol = list(df_1.columns)

#the last value of the supply column that corresponds to total value is removed
lastcol = lastcol[-1]

coltotal = {}
rowtotal = {}

#the column total of each column is stored to a dictionary
for i in df_1.columns:
    coltotal[i]=df_1[i].loc[totrows-1]

#the last value of the column total is removed as it is not required
coltotal.popitem()

#the row total of each row is stored to a dictionary
for i in range(len(df_1)):
    rowtotal[i]=df_1[lastcol].iloc[i]

#the last value of the row total is removed as it is not required
rowtotal.popitem()

#"a" is a variable defined to store the minimum value in the dataset
a = 100000000000000

#mincost is the variable used to define the cost with the minimum variable
#this variable is appended with the values of minimum variable and serves as output
mincost =0

#A while loop is set with the condition to stop the loop when the row total and coloumn total becomes zero

while (max(rowtotal.values())!=0) and (max(coltotal.values())!=0):
    #the below loop is to identify the minimum value from the dataset
    #it takes the column names and checks the total corresponding to the name in the dictionary
    #if the column total value is not zero it proceeds with the row total of different rows of the same column
    #if the column total of any column or row total of any row is found as zero, it skips it
    #the minimum values are identified only for the values where row total and column total is not zero
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
    #once the minimum value is identified, it checks whether the corresponding row total or column total is high
    #it then reduces the higher total value with the lower total value and replaces it in the corresponding dictionary
    #in the place of the higher total value
    #And the lower total value is now updated as zero
    #the minimum cost is computed by multiplying the "a" variable value with minimum total value
    #this cost is appended to the minimum cost variable and continued till all the row total and col total is zero
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

#the minimum cost variable is then printed
print("The Solution of the problem using Least Cost Method is",mincost)

