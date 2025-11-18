import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import random
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from datetime import datetime
import json
import argparse

import dataset
import model as md
import utils
   

def training(epoch_training, model,criterion,opt,schedule, data_train,data_val,track_learning_rate, tracking,track_loss, track_accuracy, freeze):
    for epoch in range(epoch_training):
        if freeze:
            #Freeze the resnet
            #train new parts of the model
            for i, child in enumerate(model.children()):
                if i == 0:
                    for parm in child.parameters():
                        parm.requires_grad = False
        else:
            for i, child in enumerate(model.children()):
                for parm in child.parameters():
                    parm.requires_grad = True
        
        model.train()
        
        
        running_loss = 0
        total = 0
        correct = 0
        for (imgs, labels) in data_train:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs,labels)
            running_loss += loss.item() * imgs.size(0)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data,1)
            correct += (predicted==labels).sum().item()
            loss.backward()
            opt.step()
        
        track_learning_rate.append(opt.param_groups[0]['lr'])
        schedule.step()
        
        avg_train_loss = running_loss / total
        track_train_loss.append(avg_train_loss)
        accuracy = correct / total * 100
        track_train_accuracy.append(accuracy)
        
         ###START Track Individual validaiton loss/accuracy vs epoch
        if tracking:
             model.eval()
             
             val_loss = 0.0
             correct = 0
             total = 0
         
             # Disable gradient calculation for validating
             with torch.no_grad():
                 for data, target in data_val:
                     # Move data and target to GPU if available
                     #print(f"target = {target}")
                     data =data.to(device)
                     target = target.to(device)
                     
                     
                     # Forward pass
                     output = model(data)
                     
                     # Calculate loss
                     loss = criterion(output, target)
                     val_loss += loss.item() * data.size(0)  # sum up batch loss
                     
                     # Get the predicted class (the one with the highest output value)
                     _, predicted = torch.max(output, 1)
                     
                     # Count correct predictions
                     correct += (predicted == target).sum().item()
                     total += target.size(0)
         
             # Calculate average loss and accuracy
             avg_loss = val_loss / total
             track_loss.append(avg_loss)
             accuracy = correct / total * 100
             track_accuracy.append(accuracy)

def testing(model, criterion, k_true, k_pred, k_acc,k_tracking, data_val, fname):
    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    true_arr = []
    pred_arr = []
    # Disable gradient calculation for valing
    with torch.no_grad():
        for data, target in data_val:
            # Move data and target to GPU if available
            true_arr.extend(target.cpu())
            data =data.to(device)
            target = target.to(device)
            
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)  # sum up batch loss
            
            # Get the predicted class (the one with the highest output value)
            _, predicted = torch.max(output, 1)
            pred_arr.extend(predicted.cpu())
            
            # Count correct predictions
            correct += (predicted == target).sum().item()
            total += target.size(0)
            if k_tracking:
                k_pred.extend(predicted.cpu())
                k_true.extend(target.cpu())

    # Calculate average loss and accuracy
    avg_loss = val_loss / total
    accuracy = correct / total * 100
    if k_tracking:
        k_acc.append(accuracy)
    utils.plot_confusion(true_arr, pred_arr, ROUTES, label_index,directory=PATH, name = fname, write_report=True)


def prototypical_loss(support_emb, support_labels, query_emb, query_labels, k, get_predicted = False):
    """
    Compute prototypical loss and accuracy for one episode.
    """
    # Step 1: compute prototypes by averaging embeddings per class
    prototypes = []
    for cls in range(size_classes):
        cls_emb = support_emb[support_labels == cls]
        proto = cls_emb.mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)  # [N, D]

    # Step 2: compute distances from queries to prototypes
    # query_emb: [N*q, D], prototypes: [N, D]
    dists = torch.cdist(query_emb, prototypes)  # [N*q, N]

    # Step 3: compute log-probabilities
    log_p_y = F.log_softmax(-dists, dim=1)  # negative distance = similarity

    # Step 4: loss
    loss = F.nll_loss(log_p_y, query_labels)

    # Step 5: accuracy
    y_hat = log_p_y.argmax(dim=1)
    acc = (y_hat == query_labels).float().mean().item()
    
    if get_predicted:
        return loss, acc, y_hat
    else:
        return loss, acc

def sample_episode(df, k, q):
    '''
    Sample a full-class episode (size_classes-way, k-shot, q-query)
    from a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A dataset that returns (image, label).
    size_classes : int
        Total number of classes in the dataset (or maximum to use).
    k : int
        Number of support examples per class (k-shot).
    q : int
        Number of query examples per class.

    Returns
    -------
    support_labels : torch.Tensor
        Shape [size_classes*k], labels for support set.
    support_images : torch.Tensor
        Shape [size_classes*k, C, H, W], support set images.
    query_labels : torch.Tensor
        Shape [size_classes*q], labels for query set.
    query_images : torch.Tensor
        Shape [size_classes*q, C, H, W], query set images.
    '''
    df_support = pd.DataFrame()
    df_query = pd.DataFrame()
    
    for l in df["Label"].unique():
        df_temp = df[df["Label"] == l]
        setnums = df_temp['Set'].unique()
    
        setnums_support = np.random.choice(setnums, k, replace=False)
        setnums = np.setdiff1d(setnums, setnums_support)
    
        setnums_query = np.random.choice(setnums, q, replace=False)
    
        df_support = pd.concat([df_support, df_temp[df_temp['Set'].isin(setnums_support)]])
        df_query   = pd.concat([df_query,   df_temp[df_temp['Set'].isin(setnums_query)]])
    
    data_support = dataset.MultiImage.create_dataloader(df_support, ROUTES,args.magnification, mode_arr, 'train', len(df_support['Set'].unique()), num_workers)
    data_query = dataset.MultiImage.create_dataloader(df_query,  ROUTES, args.magnification, mode_arr,'train', len(df_val['Set'].unique()),num_workers)
    
    support_label, support = [],[]
    for (imgs, label) in data_support:
        support_label.append(label)
        support.append(imgs)
    support = torch.cat(support,dim=0)          # shape: (N_support, C, H, W)
    support_label = torch.cat(support_label,dim=0)
    
        
    query_label, query = [],[]
    for (imgs, label) in data_query:
        query_label.append(label)
        query.append(imgs)
    query = torch.cat(query, dim=0)              # shape: (N_query, C, H, W)
    query_label = torch.cat(query_label,dim=0)
    
    return support_label, support, query_label, query


def proto_training(num_epochs, episodes_per_epoch, df, dfval, k, q, k2, q2, encoder, optimizer,track_loss, track_accuracy,track_learning_rate,schedule, tracking=False, device="cpu"):
    for epoch in range(num_epochs):
        encoder.train()
        for episode in range(episodes_per_epoch):
            # 1. Sample N-way, k-shot, q-query episode
            support_label, support, query_label, query = sample_episode(df, k, q)

            # Move to device
            support, query = support.to(device), query.to(device)
            support_label, query_label = support_label.to(device), query_label.to(device)

            # 2. Encode with shared encoder
            support_emb = encoder(support)
            query_emb   = encoder(query)

            # 3. Compute prototypical loss
            loss, acc = prototypical_loss(support_emb, support_label, query_emb, query_label, k)

            # 4. Backprop + update encoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        track_learning_rate.append(optimizer.param_groups[0]['lr'])
        schedule.step()
        
        # ---------------- VALIDATION ---------------- #
        if tracking:
            encoder.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for episode in range(episodes_per_epoch):  # run same number of val episodes
                    val_support_label, val_support, val_query_label, val_query = sample_episode(dfval, k2,q2)

                    # Move to device
                    val_support, val_query = val_support.to(device), val_query.to(device)
                    val_support_label, val_query_label = val_support_label.to(device), val_query_label.to(device)

                    # Encode
                    val_support_emb = encoder(val_support)
                    val_query_emb   = encoder(val_query)

                    # Loss + acc
                    v_loss, v_acc = prototypical_loss(val_support_emb, val_support_label, val_query_emb, val_query_label, k)

                    val_loss += v_loss.item()
                    correct  += v_acc * len(val_query_label)  # v_acc is mean acc over episode
                    total    += len(val_query_label)

            avg_loss = val_loss / episodes_per_epoch
            avg_acc = (correct / total) * 100

            track_loss.append(avg_loss)
            track_accuracy.append(avg_acc)

            
def proto_testing(episodes_per_epoch, df, k, q, encoder,k_acc, track_accuracy,k_tracking,k_pred,k_true, ROUTES, label_index,PATH, fname, device):
    encoder.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    true_arr = []
    pred_arr = []
    with torch.no_grad():
        for episode in range(episodes_per_epoch):  # run same number of val episodes
            val_support_label, val_support, val_query_label, val_query = sample_episode(df, k,q)

            # Move to device
            val_support, val_query = val_support.to(device), val_query.to(device)
            val_support_label, val_query_label = val_support_label.to(device), val_query_label.to(device)
            
            #print("device:", device)
            #print("val_support.device:", val_support.device)
            #print("encoder device:", next(encoder.parameters()).device)
            
            # Encode
            val_support_emb = encoder(val_support)
            val_query_emb   = encoder(val_query)

            # Loss + acc
            v_loss, v_acc, predicted = prototypical_loss(val_support_emb, val_support_label, val_query_emb, val_query_label, k,True)
            
            true_arr.extend(val_query_label.cpu())
            pred_arr.extend(predicted.cpu())
            
            val_loss += v_loss.item()
            correct  += v_acc * len(val_query_label)  # v_acc is mean acc over episode
            total    += len(val_query_label)
            if k_tracking:
                k_pred.extend(predicted.cpu())
                k_true.extend(val_query_label.cpu())
                

    avg_acc = (correct / total) * 100
    #print(f"avg_acc = {avg_acc}")
    if k_tracking:
        k_acc.append(avg_acc)
    utils.plot_confusion(true_arr, pred_arr, ROUTES, label_index,directory=PATH, name = fname, write_report=True)
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr1","--learningrateone",  help="Learning Rate of the newly appended model's network", type=float, default=3e-5)
    parser.add_argument("-fe","--frozenepochs", help="Number of frozen epoch training",type=int,default=1)
    parser.add_argument("-w","--workernumber", help="Number of CPU cores", type = int, default=13)
    parser.add_argument("-lr2","--learningratetwo", help="Fine Tuning Learning Rate of the entire model", type=float, default=1e-5)
    parser.add_argument("-ue","--unfrozenepochs", help="Number of unfrozen epoch training",type=int,default=10)
    parser.add_argument("-ldr", "--learningdecayrate", help="Learning rate decay rate", type=float, default = 1)
    parser.add_argument("-lds", "--learningdecaystep", help="Step Size used for decay rate... only used for Step Decay", type=float, default = 1)
    parser.add_argument("-ldr2", "--learningdecayrate2", help="Learning rate decay rate", type=float, default = 1)
    parser.add_argument("-lds2", "--learningdecaystep2", help="Step Size used for decay rate... only used for Step Decay", type=float, default = 1)
    parser.add_argument("-d", "--dropout", help = "Dropout of the network, values from 0 to 1", type=float, default=0.0)
    parser.add_argument("-wd", "--weightdecay", help = "Weightdecay", type=float, default=1e-4)
    parser.add_argument("-m", "--momentum", help = "Momentum", type=float, default=0.9)
    parser.add_argument('-k', "--kfolds", help="number of folds in the kfold validation", type=int,default =1)
    parser.add_argument('-bse', "--backscatter",help="Image mode can be SE or BSE", type=bool, default=False)
    parser.add_argument('-mag',"--magnification", help="magnficiation to be used", nargs='+', type = int, default =[10000, 50000, 100000])
    parser.add_argument('-ds', "--dataset", help="Calcination compared to other oxides", choices = ["CalcSeperate", "Calcination", "Other"], default = "CalcSeperate")
    parser.add_argument('-nt', "--normalizevalset", help = "Normalize the data set for validation", type = bool, default = False)
    parser.add_argument('-nc', "--normalizeconfusionmatrix", help="Normalize the confusion matrix", type = bool, default = False)
    parser.add_argument('-adam', "--adamopti", help = "If you want to use adam vs sgd", type = bool, default = False)
    parser.add_argument('-ts', "--trainsets", help = "What datasets to train on", nargs ='+', type = str, default = ['500-Dry'])
    parser.add_argument('-p', "--protonet", help = "If you want to use protonet vs trad classification", type = bool, default = False)
    parser.add_argument("-en","--embeddingNum", help="EmbeddingNum per image ie. final embedding will be this variable multiplied by len(mode)*len(mag)",type=int,default=64)
    parser.add_argument("-ee","--episode_per_epoch", help="Episodes per epoc",type=int,default=10)
    parser.add_argument("-et","--episode_per_training", help="Episodes per testing",type=int,default=100)
    parser.add_argument("-Kway","--Kway", help="Number of examples to be prototyped",type=int,default=3)
    parser.add_argument("-Qway","--Qway", help="Number of Queries to be tested on",type=int,default=7)
    

    
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    if args.dataset == "Calcination":
        ROUTES =["U-metal", "500-Dry", "500-Humid","700-Dry", "700-Humid","900-Dry", "900-Humid"]
        size_classes = 7
    else:
        ROUTES =["U-metal", "UO4", "AUC", "ADU", "SDU", "MDU"]
        size_classes = 6
    
    label_index = {}
    for i,r in enumerate(ROUTES):
        label_index[r]=i
    
    #if args.kfolds ==1:
    #    tracking = True
    #else:
    #    tracking = False
    tracking = True
   
    t1 = datetime.now()
    print(f"t1 = {t1}")
    print('creating output dirs')

    data_calc_dir = r"../images"
    data_other_oxide_dir = r"../otherimages"
    #model_dir = r"./model"
    output_dir = r"./output"
    
    
    model_numbers = ['model1', 'model2', 'model3', 'model4', 'model5']
    #if not os.path.exists(model_dir):
        #os.mkdir(model_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for mod in model_numbers:
        if not os.path.exists(os.path.join(output_dir,mod)):
            os.mkdir(os.path.join(output_dir,mod))
    
    print('checking for cuda')
    
    with open(os.path.join(output_dir,'argsparse.txt'), 'w') as f:
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = args.workernumber
    epoch_froze =  args.frozenepochs 
    epoch_unfroze = args.unfrozenepochs 
    kfold = args.kfolds
    if args.backscatter:
        mode_arr = ['SE', 'BSE']
    else:
        mode_arr = ['SE']
    
    PATH = os.path.join(output_dir,model_numbers[0])

    
    
    lr1 = args.learningrateone
    lr2 = args.learningratetwo
    
    k_pred = []
    k_true = []
    k_acc = []
    
    
    k_pred_train = []
    k_true_train = []
    k_acc_train = []

    
    for k in range(kfold):
        print(f'kfold = {k+1}')
        PATH = os.path.join(output_dir,model_numbers[k])
        print('creating test train split')
        
        
        if args.dataset == "CalcSeperate":
            df_train, df_val, df_test, trainsize, valsize, testsize = utils.train_val_split_multiMag(dataset = args.dataset,
                                                                                                     directory_calc = data_calc_dir,
                                                                                                     directory_otheroxide = data_other_oxide_dir,
                                                                                                     destiny = PATH,
                                                                                                     train_sets = args.trainsets,
                                                                                                     normalize_val_set = args.normalizevalset,
                                                                                                     percent=0.2,
                                                                                                     magnification = args.magnification,
                                                                                                     mode = mode_arr)
        else:
            df_train, df_val, trainsize, valsize = utils.train_val_split_multiMag(dataset = args.dataset,
                                                                                 directory_calc = data_calc_dir,
                                                                                 directory_otheroxide = data_other_oxide_dir,
                                                                                 destiny = PATH,
                                                                                 train_sets = args.trainsets,
                                                                                 normalize_val_set = args.normalizevalset,
                                                                                 percent=0.2,
                                                                                 magnification = args.magnification,
                                                                                 mode = mode_arr)
        
        
        if args.protonet:
            model = md.ProtoNet( num_mags_modes = len(args.magnification)*len(mode_arr), dropout = args.dropout, embeddingNum = args.embeddingNum).to(device)
        else:
            data_train = dataset.MultiImage.create_dataloader(df_train, ROUTES,args.magnification, mode_arr, 'train', len(df_train['Set'].unique()), num_workers)
            data_val = dataset.MultiImage.create_dataloader(df_val,  ROUTES, args.magnification, mode_arr,'val', len(df_val['Set'].unique()),num_workers)
            model = md.MultiMagCNN(num_classes = size_classes, num_mags_modes = len(args.magnification)*len(mode_arr), dropout = args.dropout).to(device)
            criterion = nn.CrossEntropyLoss()
    
        
        if args.adamopti:
            opt1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = lr1, weight_decay = args.weightdecay)
            opt2 = torch.optim.Adam(model.parameters(), lr = lr2, weight_decay = args.weightdecay)
            schedule1 = torch.optim.lr_scheduler.StepLR(opt1, step_size = args.learningdecaystep, gamma=args.learningdecayrate,verbose=False)
            schedule2 = torch.optim.lr_scheduler.StepLR(opt2, step_size = args.learningdecaystep2, gamma=args.learningdecayrate2,verbose=False)
        else:
            opt1 = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr = lr1,momentum=0.9, weight_decay = args.weightdecay)
            opt2 = torch.optim.SGD(model.parameters(), lr = lr2,momentum=0.9, weight_decay = args.weightdecay)
            schedule1 = torch.optim.lr_scheduler.StepLR(opt1, step_size = args.learningdecaystep, gamma=args.learningdecayrate,verbose=False)
            schedule2 = torch.optim.lr_scheduler.StepLR(opt2, step_size = args.learningdecaystep2, gamma=args.learningdecayrate2,verbose=False)
        
        track_accuracy = []
        track_loss = []
        track_train_accuracy = []
        track_train_loss = []
        track_learning_rate = []
    
        
        print('starting training')
        if not args.protonet:
            training(epoch_froze,model,criterion,opt1,schedule1, data_train,data_val,track_learning_rate,tracking,track_loss,track_accuracy,True)
            training(epoch_unfroze,model,criterion,opt2, schedule2, data_train,data_val,track_learning_rate,tracking,track_loss,track_accuracy,False)
        else:
            proto_training(epoch_froze, args.episode_per_epoch, df_train, df_val, args.Kway, args.Qway, 1, 2, model, opt1,track_loss, track_accuracy,track_learning_rate,schedule1, True, device)

                
        if tracking:
            fig, ax = plt.subplots(2,1, figsize = (10,20))
            #ax[0].plot(track_accuracy,label="val")
            #ax[0].plot(track_train_accuracy,label="Train")
            #ax[0].set_ylabel('Accuracy')
    
            
            ax[1].plot(track_loss,label="val")
            ax[1].plot(track_train_loss,label="Train")
            ax[1].set_ylabel('Loss')
            ax[1].set_xlabel('Epoch')
            #ax[2].set_yscale('log')
            
            #ax[2].plot(track_learning_rate,track_val_loss_for_learning_rate,'.',label="val")
            #ax[2].plot(track_learning_rate,track_loss_for_learning_rate,'.', label="Train")
            ax[0].plot(track_learning_rate,track_loss,'.',label="val")
            #ax[0].plot(track_learning_rate,track_train_loss,'.', label="Train")
            
            ax[0].set_ylabel('Loss')
            ax[0].set_xlabel('Learning Rate')
            ax[0].set_xscale('log')
            #ax[2].set_yscale('log')
    
            plt.tight_layout()
            plt.savefig(os.path.join(PATH,f"model{1}.png"),dpi=300)
            plt.close()
    
        ###END Tracking if statements
        torch.save(model.state_dict(), os.path.join(PATH, f"model{k}.pt"))
    
        if not args.protonet:
            testing(model,criterion,k_true_train,k_pred_train,k_acc_train,True, data_train,f"train-confusion{k+1}.png")
            testing(model,criterion,k_true,k_pred,k_acc,data_val,True, f"val-confusion{k+1}.png")
        else:
            proto_testing(args.episode_per_training, df_val, 1,valsize-1, model,k_acc, track_accuracy,True,k_pred,k_true,  ROUTES, label_index,PATH,f"val-confusion{k+1}.png", device)

        
        
        if args.dataset == "CalcSeperate":
            test_labels = df_test['Label'].unique()
            if not args.protonet:
                # Calculate average loss and accuracy
                for l in test_labels:
                    dft = df_test[df_test['Label']==l].copy()
                    dft['Label'] = 'U-metal'
                    data_test = dataset.MultiImage.create_dataloader(dft,  ROUTES, args.magnification, mode_arr,'test', len(dft['Set'].unique()),num_workers)
                    testing(model,criterion,None,None,None,False,data_test, f"{l}-test-confusion{k+1}.png")
            else:
                data_train = dataset.MultiImage.create_dataloader(df_train, ROUTES,args.magnification, mode_arr, 'train', len(df_train['Set'].unique()), num_workers)
                data_val = dataset.MultiImage.create_dataloader(df_val,  ROUTES, args.magnification, mode_arr,'val', len(df_val['Set'].unique()),num_workers)
                model.eval()
                with torch.no_grad():
                    support_emb, support_label = [],[]
                    for (qu, labs) in data_val:
                        qu = qu.to(device)
                        labs = labs.to(device)
                        support_emb.append(model(qu))
                        support_label.append(labs)
                    support_emb = torch.cat(support_emb,dim=0)
                    support_label = torch.cat(support_label, dim=0)
                    
                    for l in test_labels:
                        val_loss = 0
                        correct  = 0
                        total = 0
                        true_arr = []
                        pred_arr = []
                        dft = df_test[df_test['Label']==l].copy()
                        dft['Label'] = 'U-metal'
                        data_test = dataset.MultiImage.create_dataloader(dft,  ROUTES, args.magnification, mode_arr,'test', len(dft['Set'].unique()),num_workers)
                        query_emb, query_label = [],[]
                        for (dat, labs) in data_test:
                            dat = dat.to(device)
                            labs = labs.to(device)
                            query_emb.append(model(dat))
                            query_label.append(labs)
                        
                        query_emb= torch.cat(query_emb,dim=0)
                        query_label = torch.cat(query_label,dim=0)
                        

                        # Loss + acc
                        v_loss, v_acc, predicted = prototypical_loss(support_emb, support_label, query_emb, query_label,  k,True)
                        true_arr.extend(query_label.cpu())
                        pred_arr.extend(predicted.cpu())
                        
                        
                        val_loss += v_loss.item()
                        correct  += v_acc * len(query_label)  # v_acc is mean acc over episode
                        total    += len(query_label)
                        
                        avg_acc = (correct / total) * 100
                        #print(f"avg_acc = {avg_acc}")
                        utils.plot_confusion(true_arr, pred_arr, ROUTES, label_index,directory=PATH, name = f"{l}-test-confusion-{k+1}.png", write_report=True)
                    
    #utils.plot_confusion(k_true_train, k_pred_train, ROUTES, label_index, directory=output_dir, name = "train-confusionALL.png", write_report=True)
    utils.plot_confusion(k_true, k_pred, ROUTES, label_index,directory=output_dir, name = "val-confusionALL.png", write_report=True)
    t2 = datetime.now()
    print(f"Finished\nt2 = {t2}")
