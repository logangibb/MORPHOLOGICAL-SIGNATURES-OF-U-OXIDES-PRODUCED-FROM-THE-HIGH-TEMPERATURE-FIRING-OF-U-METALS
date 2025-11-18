import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import torch
import math
from PIL import Image
from statsmodels.stats.proportion import proportion_confint


def plot_confusion(true, pred, routes, label_index, directory=None, name = None, write_report=True):
    true = np.array(true)
    pred = np.array(pred)
    
    true_size = len(true)
    pred_size = len(pred)
    
    routes_size = len(routes)
    if true_size != pred_size:
        print('EXPECTED 2 ARRAYS OF THE SAME SIZE')
    answer = np.zeros((routes_size,routes_size))
    for rrow in routes:
        lrow = label_index[rrow]
        new_arr = pred[true==lrow]
        for rcol in routes:
            lcol = label_index[rcol]
            answer[lrow,lcol] = np.sum(np.where(new_arr==lcol,1,0))
    
    
    row_sums = answer.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    accuracies = answer / row_sums
    
    fig, ax = plt.subplots()
    cb = ax.imshow(accuracies, cmap='Blues')
    
    ax.set_xticks(np.arange(routes_size), labels=routes)
    ax.set_yticks(np.arange(routes_size), labels=routes)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    
    
    fmt = ".0f"
    
    for i in range(len(routes)):
        for j in range(len(routes)):
            x = answer[i,j]
            
            if accuracies[i,j] < 0.5:
                color = 'blue'
            else:
                color = 'white'
            ax.text(j, i, format(x, fmt), ha="center", va="center", color=color)
            
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar(cb, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(directory, name), dpi=300)
    plt.close()
    
    
    fig, ax = plt.subplots()
    cb = ax.imshow(accuracies, cmap='Blues')
    
    ax.set_xticks(np.arange(routes_size), labels=routes)
    ax.set_yticks(np.arange(routes_size), labels=routes)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    
    
    fmt = ".2f" 
    
    for i in range(len(routes)):
        for j in range(len(routes)):
            x = accuracies[i,j]
            
            if accuracies[i,j] < 0.5:
                color = 'blue'
            else:
                color = 'white'
            ax.text(j, i, format(x, fmt), ha="center", va="center", color=color)
            
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar(cb, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(directory, "normalized" + name), dpi=300)
    plt.close()
    
    # ---------- Write metrics report ----------
    if write_report:
        report_path = os.path.join(directory, "report_" + name.replace(".png", ".txt"))
        with open(report_path, "w") as f:
            f.write("Classification Report\n")
            f.write("="*50 + "\n\n")
            
            total_correct = 0
            total_samples = np.sum(answer)
            
            for i, cls in enumerate(routes):
                tp = answer[i, i]
                fn = answer[i, :].sum() - tp
                fp = answer[:, i].sum() - tp
                tn = total_samples - (tp + fn + fp)

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                # Per-class accuracy & Wilson CI
                total = tp + fn
                acc = tp / total if total > 0 else 0
                low, high = proportion_confint(tp, total, method="wilson") if total > 0 else (0, 0)

                f.write(f"Class: {cls}\n")
                f.write(f"  Precision: {prec:.3f}\n")
                f.write(f"  Recall:    {rec:.3f}\n")
                f.write(f"  F1-score:  {f1:.3f}\n")
                f.write(f"  Accuracy:  {acc:.3f}\n")
                f.write(f"  95% CI (Wilson): [{low:.3f}, {high:.3f}]\n\n")

                total_correct += tp
            
            # Overall accuracy
            overall_acc = total_correct / total_samples
            f.write("="*50 + "\n")
            f.write(f"Overall Accuracy: {overall_acc:.3f}\n")
            f.write("="*50 + "\n")
        
        cm_path = os.path.join(directory, "confusion_matrix_" + name.replace(".png", ".txt"))
        np.savetxt(cm_path, answer, fmt="%d")
    
    
   

    




def train_val_split_multiMag(dataset, directory_calc, directory_otheroxide,destiny,train_sets, normalize_val_set = False,percent=0.2, magnification = [10000], mode = ['SE']):
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    setnum = 1


    drop_arr = ["image 111.png", "image 112.png", "image 113.png", "image 114.png", "image 115.png"]
    if dataset == "CalcSeperate":
        #print("Other")
        dfotheroxide = pd.read_csv(os.path.join(directory_otheroxide,"otheroxides.csv")) #read in the csv with the file names and labels
        dfotheroxide['directory'] = directory_otheroxide
        dfotheroxide["Mag"] = dfotheroxide["Magnification"].str.split("x").str[0].astype(int)
        dfotheroxide['OriginalFile'] = dfotheroxide['FileName']
        dfcalc = pd.read_csv(os.path.join(directory_calc,"calc.csv"))
        
        dfcalc['directory'] = directory_calc
        dfcalc = dfcalc[dfcalc["Label"] != "U-metal"]
        df = dfotheroxide.copy()
        for ts in train_sets:
            df_temp = dfcalc[dfcalc['Label'] == ts].copy()
            dfcalc = dfcalc.drop(df_temp.index)
            df_temp['Label'] = 'U-metal'
            df = pd.concat([df,df_temp], ignore_index=True)
        
        
        dfcalc['Set']=0
        
        for mag in df["Mag"].unique():
            if mag not in magnification:
                dfcalc = dfcalc[dfcalc['Mag'] != mag]
                
                
        labels = dfcalc['Label'].unique()
        minsize = 100000
        size_arr = []
        for l in labels:
            df_label = dfcalc[dfcalc["Label"]==l].copy()
            size = 0
            for m in mode:
                for mag in magnification:
                    size = np.sum((df_label['Mode']==m)*(df_label['Mag']==mag))//4
                    size_arr.append(size)
                    if minsize > size:
                        minsize = size
            
        size_arr = np.array(size_arr)
        
        testsize = minsize
        
        for l in labels:
            df_label = dfcalc[dfcalc["Label"]==l].copy()
            for g in df_label['Group'].unique():
                if np.sum(df_label['Group']==g) < len(magnification)*len(mode)*4:
                    dfcalc = dfcalc.drop(df_label[df_label['Group']==g].index, axis=0)

        for l in labels:
            df_label = dfcalc[dfcalc["Label"]==l].copy()
            df_group = df_label['Group'].unique()
            dfindex = np.random.choice(df_group,testsize,False)
            for g in dfindex:
                df_test = pd.concat([df_test,dfcalc[dfcalc['Group']==g]])

        for l in labels:
            for m in mode:
                for mag in magnification:
                    df_temp = df_test[(df_test['Label']==l)*(df_test['Mode']==m)*(df_test['Mag']==mag)].copy()
                    set_arr = np.arange(setnum,setnum+len(df_temp),1)
                    df_test.loc[(df_test['Label']==l)*(df_test['Mode']==m)*(df_test['Mag']==mag),['Set']] = set_arr
            setnum += len(df_temp)
         

        

        for s in df_test['Set'].unique():
            len_s  = int(np.sum(df_test['Set']==s))
            if len_s != 6:
                df_test = df_test[df_test['Set'] != s]
        df_test.to_csv(os.path.join(destiny,"test.csv"),index=False)
           
    elif dataset == "Calcination":
        #print("Calcination")
        df = pd.read_csv(os.path.join(directory_calc,"calc.csv")) #read in the csv with the file names and labels
        df['directory'] = directory_calc
        for d in drop_arr:
            df = df[df["file"]!=d]
    else:
        #print("Other")
        dfotheroxide = pd.read_csv(os.path.join(directory_otheroxide,"otheroxides.csv")) #read in the csv with the file names and labels
        dfotheroxide['directory'] = directory_otheroxide
        dfotheroxide["Mag"] = dfotheroxide["Magnification"].str.split("x").str[0].astype(int)
        dfotheroxide['OriginalFile'] = dfotheroxide['FileName']
        dfcalc = pd.read_csv(os.path.join(directory_calc,"calc.csv"))
        for d in drop_arr:
            dfcalc = dfcalc[dfcalc["file"]!=d]
        dfcalc = dfcalc[dfcalc["Label"] != "U-metal"]
        dfcalc['directory'] = directory_calc
        dfcalc['Label'] = 'U-metal'
        df = pd.concat([dfotheroxide,dfcalc], ignore_index=True)



    df['Set']=0

    for mag in df["Mag"].unique():
        if mag not in magnification:
            df = df[df['Mag'] != mag]
            
            
    labels = df['Label'].unique()
    minsize = 100000
    size_arr = []
    for l in labels:
        df_label = df[df["Label"]==l].copy()
        size = 0
        for m in mode:
            for mag in magnification:
                size = np.sum((df_label['Mode']==m)*(df_label['Mag']==mag))//4
                size_arr.append(size)
                if minsize > size:
                    minsize = size
        
    size_arr = np.array(size_arr)
    trainsize = round(minsize*(1-percent))
    valsize = round(minsize*percent)
            
    for l in labels:
        df_label = df[df["Label"]==l].copy()
        for g in df_label['Group'].unique():
            if np.sum(df_label['Group']==g) < len(magnification)*len(mode)*4:
                df = df.drop(df_label[df_label['Group']==g].index, axis=0)

    for l in labels:
        df_label = df[df["Label"]==l].copy()
        df_group = df_label['Group'].unique()
        dfindex = np.random.choice(df_group,trainsize,False)
        for g in dfindex:
            df_train = pd.concat([df_train,df[df['Group']==g]])
        
        antidf = df_group[~np.isin(df_group,dfindex)]
        if normalize_val_set:
            antidfsample = np.random.choice(antidf,valsize,False)
            for g in antidfsample:
                df_val = pd.concat([df_val,df[df['Group']==g]])
        else:
            for g in antidf:
                df_val = pd.concat([df_val,df[df['Group']==g]])


    for l in labels:
        for m in mode:
            for mag in magnification:
                df_temp = df_train[(df_train['Label']==l)*(df_train['Mode']==m)*(df_train['Mag']==mag)].copy()
                set_arr = np.arange(setnum,setnum+len(df_temp),1)
                df_train.loc[(df_train['Label']==l)*(df_train['Mode']==m)*(df_train['Mag']==mag),['Set']] = set_arr
        setnum += len(df_temp)


    for l in labels:
        for m in mode:
            for mag in magnification:
                df_temp = df_val[(df_val['Label']==l)*(df_val['Mode']==m)*(df_val['Mag']==mag)].copy()
                set_arr = np.arange(setnum,setnum+len(df_temp),1)
                df_val.loc[(df_val['Label']==l)*(df_val['Mode']==m)*(df_val['Mag']==mag),['Set']] = set_arr
        setnum += len(df_temp)        


    for s in df_train['Set'].unique():
        len_s  = np.sum(df_train['Set']==s)
        if len_s != 6:
            df_train = df_train[df_train['Set'] != s]
            #print(f"{s}; {len_s}")

    for s in df_val['Set'].unique():
        len_s  = np.sum(df_val['Set']==s)
        if len_s != 6:
            df_val = df_val[df_val['Set'] != s]
            #print(f"{s}; {len_s}")    


    df_train.to_csv(os.path.join(destiny,"train.csv"),index=False)
    df_val.to_csv(os.path.join(destiny,"val.csv"), index=False)
    df_test.to_csv(os.path.join(destiny,"test.csv"),index=False)
    if dataset == "CalcSeperate":
        return df_train,df_val,df_test, trainsize, valsize, testsize
    else:
        return df_train, df_val, trainsize, valsize

'''
#for multi mag
dataset = "CalcSeperate"
directory_calc = "C:\\Users\\Logan\\Box\\Logan Metal ML\\calcination\\images"
directory_otheroxide = "C:\\Users\\Logan\\Box\\Logan Metal ML\\calcination\\UoxideComparisionCropped"
destiny = "C:\\Users\\Logan\\Box\\Logan Metal ML\\calcination\\images"
percent = 0.2
magnification = [10000, 50000, 100000]
mode = ["SE", "BSE"]
normalize_val_set = False
train_sets = ['500-Dry']




df_test = pd.DataFrame()
df_train = pd.DataFrame()
df_val = pd.DataFrame()
setnum = 1


drop_arr = ["image 111.png", "image 112.png", "image 113.png", "image 114.png", "image 115.png"]
if dataset == "CalcSeperate":
    #print("Other")
    dfotheroxide = pd.read_csv(os.path.join(directory_otheroxide,"otheroxides.csv")) #read in the csv with the file names and labels
    dfotheroxide['directory'] = directory_otheroxide
    dfotheroxide["Mag"] = dfotheroxide["Magnification"].str.split("x").str[0].astype(int)
    dfotheroxide['OriginalFile'] = dfotheroxide['FileName']
    dfcalc = pd.read_csv(os.path.join(directory_calc,"calc.csv"))
    
    dfcalc['directory'] = directory_calc
    dfcalc = dfcalc[dfcalc["Label"] != "U-metal"]
    df = dfotheroxide.copy()
    for ts in train_sets:
        df_temp = dfcalc[dfcalc['Label'] == ts].copy()
        dfcalc = dfcalc.drop(df_temp.index)
        df_temp['Label'] = 'U-metal'
        df = pd.concat([df,df_temp], ignore_index=True)
    
    
    dfcalc['Set']=0
    
    for mag in df["Mag"].unique():
        if mag not in magnification:
            dfcalc = dfcalc[dfcalc['Mag'] != mag]
            
            
    labels = dfcalc['Label'].unique()
    minsize = 100000
    size_arr = []
    for l in labels:
        df_label = dfcalc[dfcalc["Label"]==l].copy()
        size = 0
        for m in mode:
            for mag in magnification:
                size = np.sum((df_label['Mode']==m)*(df_label['Mag']==mag))//4
                size_arr.append(size)
                if minsize > size:
                    minsize = size
        
    size_arr = np.array(size_arr)
    
    testsize = minsize
    
    for l in labels:
        df_label = dfcalc[dfcalc["Label"]==l].copy()
        for g in df_label['Group'].unique():
            if np.sum(df_label['Group']==g) < len(magnification)*len(mode)*4:
                dfcalc = dfcalc.drop(df_label[df_label['Group']==g].index, axis=0)

    for l in labels:
        df_label = dfcalc[dfcalc["Label"]==l].copy()
        df_group = df_label['Group'].unique()
        dfindex = np.random.choice(df_group,testsize,False)
        for g in dfindex:
            df_test = pd.concat([df_test,dfcalc[dfcalc['Group']==g]])

    for l in labels:
        for m in mode:
            for mag in magnification:
                df_temp = df_test[(df_test['Label']==l)*(df_test['Mode']==m)*(df_test['Mag']==mag)].copy()
                set_arr = np.arange(setnum,setnum+len(df_temp),1)
                df_test.loc[(df_test['Label']==l)*(df_test['Mode']==m)*(df_test['Mag']==mag),['Set']] = set_arr
        setnum += len(df_temp)
     

    

    for s in df_test['Set'].unique():
        len_s  = int(np.sum(df_test['Set']==s))
        if len_s != 6:
            df_test = df_test[df_test['Set'] != s]
    df_test.to_csv(os.path.join(destiny,"test.csv"),index=False)
       
elif dataset == "Calcination":
    #print("Calcination")
    df = pd.read_csv(os.path.join(directory_calc,"calc.csv")) #read in the csv with the file names and labels
    df['directory'] = directory_calc
    for d in drop_arr:
        df = df[df["file"]!=d]
else:
    #print("Other")
    dfotheroxide = pd.read_csv(os.path.join(directory_otheroxide,"otheroxides.csv")) #read in the csv with the file names and labels
    dfotheroxide['directory'] = directory_otheroxide
    dfotheroxide["Mag"] = dfotheroxide["Magnification"].str.split("x").str[0].astype(int)
    dfotheroxide['OriginalFile'] = dfotheroxide['FileName']
    dfcalc = pd.read_csv(os.path.join(directory_calc,"calc.csv"))
    for d in drop_arr:
        dfcalc = dfcalc[dfcalc["file"]!=d]
    dfcalc = dfcalc[dfcalc["Label"] != "U-metal"]
    dfcalc['directory'] = directory_calc
    dfcalc['Label'] = 'U-metal'
    df = pd.concat([dfotheroxide,dfcalc], ignore_index=True)



df['Set']=0

for mag in df["Mag"].unique():
    if mag not in magnification:
        df = df[df['Mag'] != mag]
        
        
labels = df['Label'].unique()
minsize = 100000
size_arr = []
for l in labels:
    df_label = df[df["Label"]==l].copy()
    size = 0
    for m in mode:
        for mag in magnification:
            size = np.sum((df_label['Mode']==m)*(df_label['Mag']==mag))//4
            size_arr.append(size)
            if minsize > size:
                minsize = size
    
size_arr = np.array(size_arr)
trainsize = round(minsize*(1-percent))
valsize = round(minsize*percent)
        
for l in labels:
    df_label = df[df["Label"]==l].copy()
    for g in df_label['Group'].unique():
        if np.sum(df_label['Group']==g) < len(magnification)*len(mode)*4:
            df = df.drop(df_label[df_label['Group']==g].index, axis=0)

for l in labels:
    df_label = df[df["Label"]==l].copy()
    df_group = df_label['Group'].unique()
    dfindex = np.random.choice(df_group,trainsize,False)
    for g in dfindex:
        df_train = pd.concat([df_train,df[df['Group']==g]])
    
    antidf = df_group[~np.isin(df_group,dfindex)]
    if normalize_val_set:
        antidfsample = np.random.choice(antidf,valsize,False)
        for g in antidfsample:
            df_val = pd.concat([df_val,df[df['Group']==g]])
    else:
        for g in antidf:
            df_val = pd.concat([df_val,df[df['Group']==g]])


for l in labels:
    for m in mode:
        for mag in magnification:
            df_temp = df_train[(df_train['Label']==l)*(df_train['Mode']==m)*(df_train['Mag']==mag)].copy()
            set_arr = np.arange(setnum,setnum+len(df_temp),1)
            df_train.loc[(df_train['Label']==l)*(df_train['Mode']==m)*(df_train['Mag']==mag),['Set']] = set_arr
    setnum += len(df_temp)


for l in labels:
    for m in mode:
        for mag in magnification:
            df_temp = df_val[(df_val['Label']==l)*(df_val['Mode']==m)*(df_val['Mag']==mag)].copy()
            set_arr = np.arange(setnum,setnum+len(df_temp),1)
            df_val.loc[(df_val['Label']==l)*(df_val['Mode']==m)*(df_val['Mag']==mag),['Set']] = set_arr
    setnum += len(df_temp)        


for s in df_train['Set'].unique():
    len_s  = np.sum(df_train['Set']==s)
    if len_s != 6:
        df_train = df_train[df_train['Set'] != s]
        #print(f"{s}; {len_s}")

for s in df_val['Set'].unique():
    len_s  = np.sum(df_val['Set']==s)
    if len_s != 6:
        df_val = df_val[df_val['Set'] != s]
        #print(f"{s}; {len_s}")
        



for l in df_val['Label'].unique():
    print(f"{l}: {np.sum(df_val['Label']==l)}")



'''





































