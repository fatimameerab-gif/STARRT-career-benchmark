import sys,pandas as pd,numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
d=pd.read_excel(sys.argv[1]).dropna()
d['Skill']=d.Skill.astype(str).str.strip(); d['Career']=d.Career.str.strip()
print('rows',len(d),'exact dup rows',d.duplicated().sum())
same_text_diff_label=d.groupby('Skill').Career.nunique(); print('texts with >1 label:',(same_text_diff_label>1).sum())
u=d.drop_duplicates(); print('after dedup',len(u)); print(u.Career.value_counts().to_string())
cv=StratifiedKFold(5,shuffle=True,random_state=42)
pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=3000,class_weight='balanced'))
for name,df in [('with duplicates (leaky)',d),('deduplicated',u)]:
    print(name,'majority',cross_val_score(DummyClassifier(),df.Skill,df.Career,cv=cv).mean().round(3),
          'tfidf acc',cross_val_score(pipe,df.Skill,df.Career,cv=cv).mean().round(3),
          'macroF1',cross_val_score(pipe,df.Skill,df.Career,cv=cv,scoring='f1_macro').mean().round(3))
print('words per row:',u.Skill.str.split().str.len().describe().round(1).to_dict())
