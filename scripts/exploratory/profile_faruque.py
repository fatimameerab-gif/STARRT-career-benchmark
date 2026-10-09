import sys,pandas as pd
x=pd.ExcelFile(sys.argv[1]); print('sheets',x.sheet_names)
for s in x.sheet_names:
    d=x.parse(s); print('==',s,d.shape); print(d.columns.tolist())
    pd.set_option('display.max_colwidth',160); print(d.head(5).to_string())
    for c in d.columns:
        if d[c].nunique()<40: print('\n',c); print(d[c].value_counts().to_string())
    print('dup rows:',d.duplicated().sum())
