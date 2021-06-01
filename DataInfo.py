def getstatfunction(name,df):
    if name == 'Name':
        return lambda column:column
    elif name == 'Mean':
        return lambda column:df[column].mean()
    elif name == 'Median':
        return lambda column:df[column].median()
    elif name == 'Range':
        return lambda column:df[column].max()-df[column].min()
    elif name == 'Variance':
        return lambda column:df[column].var()
    elif name == 'Std. Dev':
        return lambda column:df[column].std()

def printRow(name,columnformat,df,columns):
    stat = getstatfunction(name,df)
    print("{:20s}  ".format(name), end="")
    for i in columns:
        print(columnformat.format(stat(i)), end="")
    print('')

def getDataInfo(df,columns):
    printRow('Name',"{:>20s}  ",df,columns)
    printRow('Mean',"{:20.4f}  ",df,columns)
    printRow('Median',"{:20.4f}  ",df,columns)
    printRow('Range',"{:20.4f}  ",df,columns)
    printRow('Variance',"{:20.4f}  ",df,columns)
    printRow('Std. Dev',"{:20.4f}  ",df,columns)