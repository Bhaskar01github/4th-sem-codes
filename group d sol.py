
coll = ['Rasgulla is a sweet', 'Rasmalai is also a sweet',
 'Chocolate is also sweet but there are dark chocolates too', 'Pantua and sandesh are also sweets', 
 'I love mishti doi', 'Sweets are the heart of Kolkata', 
 'In the interiors you will find sweets like monohara and sarpuria', 'You will also find biriyani in Kolkata',
 'Aminia is known for biriyani', 'Arsalan is also known for biriyani', 
 'Wajid Ali Shah is credited for bringing biriyani to Kolkata',
 'Wajid Ali Shah was a Nawab', 'Satyajit Ray made a movie named Shatranj ke Khiladi on Wajid Ali Shah', 
 'Satyajit Ray was a great filmmaker']    
def preprocess(s):
 s = s.lower()
 s = s.split(' ')
 stop = ['also','we','our','ours','ourselves','you','your','yours',
 'yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
 'they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those',
 'am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing',
 'a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
 'against','between','into','through','during','before','after','above','below','to','from','up','down',
 'in','out','on','off','over','under','again','further','then','once','here','there','when',
 'where','why','how','all', 'any','both','each','few','more','most','other','some','such',
 'no','nor','not','only','own','same','so','than',
 'too','very','s','t','can','will','just','don','should','now']
 s_out = []
 for w in s:
  if w not in stop:
   s_out.append(w)
 return s_out
newcoll=[]
for i in range(len(coll)):
 newcoll.append(preprocess(coll[i]))   
badkey=[item for sublist in newcoll for item in sublist]
key=list(set(badkey))
dict={}
a=[]
for j in key:
     a=[]   
     if j not in dict:
      for i in range(len(newcoll)):
        if j in newcoll[i]:
         for k in newcoll[i]:
             if k!=j:
              a.append(k)
      dict[j]=a  
print(dict["sweet"])  
i=input("enter a word")
for i in range()            





#PANDA

names=input('enter the names of 4 actress ').split(',')
on=input('enter the no of times nominated in oskar ' ).split(',')
ow=input('enter the no of won ').split(',')
bn=input('no of times nominated in bafta ').split(',')
bw=input('enter the no of times won ').split(',')
on=[int(i) for i in on]
ow=[int(i) for i in ow]
bn=[int(i) for i in bn]
bw=[int(i) for i in bw]
dictionary={'names':names,'on':on,'ow':ow,'bn':bn,'bw':bw}
import pandas as p
act= p.DataFrame(dictionary, index = ['a1', 'a2', 'a3', 'a4'])
print(act)
newrow=p.Series(data=['ash',0,0,0,0],index=act.columns,names='a5')
print(act)
act=act.drop(labels=['row4'])
print(act)
act('total_nom')=act['on']+act['bn']
act('total_win')=act['ow']+act['bw']
print(act.mean())
print(act('total_nom').corr('total_win'))
import matplotlib.pyplot as plt
act.plt(kind='line',x='total_nom',y='total_win',marker='o')
plt.title('comparison')
plt.show()
act.plt(kind='pie',x='total_nom')


#PANDA

names=input('enter the names of 4 actress ').split(',')
on=input('enter the no of times nominated in oskar ' ).split(',')
ow=input('enter the no of won ').split(',')
bn=input('no of times nominated in bafta ').split(',')
bw=input('enter the no of times won ').split(',')
on=[int(i) for i in on]
ow=[int(i) for i in ow]
bn=[int(i) for i in bn]
bw=[int(i) for i in bw]
dictionary={'names':names,'on':on,'ow':ow,'bn':bn,'bw':bw}
import pandas as p
act= p.DataFrame(dictionary, index = ['a1', 'a2', 'a3', 'a4'])
print(act)
newrow=p.Series(data=['ash',0,0,0,0],index=act.columns,names='a5')
print(act)
act=act.drop(labels=['row4'])
print(act)
act('total_nom')=act['on']+act['bn']
act('total_win')=act['ow']+act['bw']
print(act.mean())
print(act('total_nom').corr('total_win'))
import matplotlib.pyplot as plt
act.plt(kind='line',x='total_nom',y='total_win',marker='o')
plt.title('comparison')
plt.show()
act.plt(kind='pie',x='total_nom')


#question 3
import random as r 
samgo={1:'sam goes out ',0:'sam does not goes out'}
tomgo={1:'tom goes out ',0:'tom does not goes out'}
tomjuice={1:'tom gets apple juice',0:'tom does not get apple juice'}
samjuice={1:'sam gets mango juice',0:'sam does not get mango juice'}
tomroad={1:'tom choose the correct road',0:'tom choose the wrong road'}
samroad={1:'sam choose the correct road',0:'sam choose the wrong road'}
#lets look what sam is doing 
a1=r.random()
b1=r.random()
c1=r.uniform(0,1)
if a1<0.7:
    samgo_=1
else:
    samgo_=0
if samgo_==0:
 samroad_=0
 sam_juice=0
else:    
 if c1<0.6:
    samroad_=1
 else:
    samroad_=0
 if samroad_==0:
     sam_juice=0
 else:
  if b1<0.8:
    sam_juice=1
  else:
    sam_juice=0
# lets look what tom is doing 
a2=r.random()
b2=r.random()
c2=r.uniform(0,1)
if a2<0.7:
    tomgo_=1
else:
    tomgo_=0
if tomgo_==0:
 tomroad_=0
 tom_juice=0
else:    
 if c2<0.6:
    tomroad_=1
 else:
    tomroad_=0
 if tomroad_==0:
     tom_juice=0
 else:
  if b2<0.8:
    tom_juice=1
  else:
    tom_juice=0
print(samgo[samgo_])
print(samroad[samroad_])
print(samjuice[sam_juice])    
print(tomgo[tomgo_])
print(tomroad[tomroad_])
print(tomjuice[tom_juice]) 
SAM=[samgo_,samroad_,sam_juice]   
TOM=[tomgo_,tomroad_,tom_juice]   
if SAM==[1,1,1] and TOM==[1,1,1]:
    print('they met together')
else:
    print('they do not meet together')
    
    
#question 3
import random as r 
samgo={1:'sam goes out ',0:'sam does not goes out'}
tomgo={1:'tom goes out ',0:'tom does not goes out'}
tomjuice={1:'tom gets apple juice',0:'tom does not get apple juice'}
samjuice={1:'sam gets mango juice',0:'sam does not get mango juice'}
tomroad={1:'tom choose the correct road',0:'tom choose the wrong road'}
samroad={1:'sam choose the correct road',0:'sam choose the wrong road'}
#lets look what sam is doing 
a1=r.random()
b1=r.random()
c1=r.uniform(0,1)
if a1<0.7:
    samgo_=1
else:
    samgo_=0
if samgo_==0:
 samroad_=0
 sam_juice=0
else:    
 if c1<0.6:
    samroad_=1
 else:
    samroad_=0
 if samroad_==0:
     sam_juice=0
 else:
  if b1<0.8:
    sam_juice=1
  else:
    sam_juice=0
# lets look what tom is doing 
a2=r.random()
b2=r.random()
c2=r.uniform(0,1)
if a2<0.7:
    tomgo_=1
else:
    tomgo_=0
if tomgo_==0:
 tomroad_=0
 tom_juice=0
else:    
 if c2<0.6:
    tomroad_=1
 else:
    tomroad_=0
 if tomroad_==0:
     tom_juice=0
 else:
  if b2<0.8:
    tom_juice=1
  else:
    tom_juice=0
print(samgo[samgo_])
print(samroad[samroad_])
print(samjuice[sam_juice])    
print(tomgo[tomgo_])
print(tomroad[tomroad_])
print(tomjuice[tom_juice]) 
SAM=[samgo_,samroad_,sam_juice]   
TOM=[tomgo_,tomroad_,tom_juice]   
if SAM==[1,1,1] and TOM==[1,1,1]:
    print('they met together')
else:
    print('they do not meet together')
    
    