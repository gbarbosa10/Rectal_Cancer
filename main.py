import os
import utils
import argparse
import numpy as np
from neptune.types import File
from pytorch_lightning.tuner import Tuner
from sklearn.model_selection import ParameterGrid
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import subprocess
import pickle
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE    
import torch.nn as nn
import torchvision.models as models
import ssl
import urllib3


def get_model(model_name, dropout):
    model_weight_dict ={"regnet_y_128gf" : "RegNet_Y_128GF_Weights", "vit_h_14" : "ViT_H_14_Weights","vgg19" : "VGG19_Weights", "vgg19_bn" : "VGG19_BN_Weights",  "regnet_y_128gf" : "RegNet_Y_128GF_Weights", "resnext101_32x8d" : "ResNeXt101_32X8D_Weights", "wide_resnet101_2" : "Wide_ResNet101_2_Weights", "convnext_large" : "ConvNeXt_Large_Weights", "efficientnet_v2_l" : "EfficientNet_V2_L_Weights"}

    weights_class_name = model_weight_dict[model_name]
    weights = getattr(models, weights_class_name).DEFAULT 

    model_constructor = getattr(models, model_name.lower())
    model = model_constructor(weights=weights) if weights else model_constructor(pretrained=True)
    if model_name == "vgg19_bn":
        model.classifier[6] = nn.Linear(4096, 3)
    elif model_name == "vgg19":
        model.classifier[6] = nn.Linear(4096, 3)
    elif model_name == "resnext101_32x8d":
        model.fc = nn.Sequential(nn.Dropout(dropout), 
                                 nn.Linear(2048, 256), 
                                 #nn.ReLU(), 
                                 #nn.Linear(1024, 256), 
                                 nn.ReLU(), 
                                 nn.Linear(256, 3))
    elif model_name == "wide_resnet101_2":
        model.fc = nn.Sequential(nn.Dropout(dropout), 
                                 nn.Linear(2048, 256), 
                                 #nn.ReLU(), 
                                 #nn.Linear(1024, 256), 
                                 nn.ReLU(), 
                                 nn.Linear(256, 3))
    elif model_name == "convnext_large":
        model.classifier[2] = nn.Linear(1536, 3)
    elif model_name == "efficientnet_v2_l":
        model.classifier = nn.Sequential(nn.Dropout(dropout), 
                                 nn.Linear(1280, 128), 
                                 #nn.ReLU(), 
                                 #nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 3))
    elif model_name == "regnet_y_128gf":
        model.fc = nn.Linear(7392, 3)
    elif model_name == "vit_h_14":
        model.heads.head = nn.Linear(in_features=1280, out_features=3)
    return model

def get_cnn_pretrained(model_name):
    
    model_weight_dict ={"regnet_y_128gf" : "RegNet_Y_128GF_Weights", "vit_h_14" : "ViT_H_14_Weights","vgg19" : "VGG19_Weights", "vgg19_bn" : "VGG19_BN_Weights",  "regnet_y_128gf" : "RegNet_Y_128GF_Weights", "resnext101_32x8d" : "ResNeXt101_32X8D_Weights", "wide_resnet101_2" : "Wide_ResNet101_2_Weights", "convnext_large" : "ConvNeXt_Large_Weights", "efficientnet_v2_l" : "EfficientNet_V2_L_Weights"}

    weights_class_name = model_weight_dict[model_name]
    weights = getattr(models, weights_class_name).DEFAULT 

    model_constructor = getattr(models, model_name.lower())
    model = model_constructor(weights=weights) if weights else model_constructor(pretrained=True)
    return model

def get_feature_extractor(model):
    print(model)
    if hasattr(model, 'features'):
        return nn.Sequential(*list(model.features.children()))  # Extract features for models like VGG, MobileNetV2
    if hasattr(model, 'heads'):
        model.heads.head = nn.Linear(in_features=1280, out_features=3)
        return nn.Sequential(*list(model.children()))  # Extract features for models like VGG, MobileNetV2
    else:
        # For models like ResNet and EfficientNet, remove the classifier (last fully connected layer)
        return nn.Sequential(*list(model.children())[:-1])

def get_classifier(model):
    classifier_attrs = ['fc', 'classifier', 'head', 'heads']
    
    for attr in classifier_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)
    
    raise AttributeError("The classifier/head was not found in the model.")

def change_first_layer_classifier(model, n_neurons):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            parent_module = model
            *path, last_name = name.split('.')
            for part in path:
                parent_module = getattr(parent_module, part)
            in_features = module.in_features
            new_linear = nn.Linear(in_features, n_neurons)
            setattr(parent_module, last_name, new_linear)
            break
    return model


torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()

idx_to_cls_names = {0: "Normal", 1: "Complete Response", 2: "Residual Recurrence"}
n_classes = len(idx_to_cls_names.keys())
parameter_grid = {"Learning Rate": [0.00001, 0.000001, 0.001],
              "Batch Size": [8, 16, 32, 64],
              "Dropout": [0.7],
              "Models": ["wide_resnet101_2", "convnext_large", "vgg19", "vgg19_bn", "vit_h_14"],
              "Fold": [1,3],
              "Weighted Random Sampler": [True]}
pat = 25; data_aug_key = "all"

if __name__ == "__main__":
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    print(f"Using {device} device")

    ssl.get_default_verify_paths()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(42)
    g = torch.Generator()
    g.manual_seed(42)

    rootdir = os.path.join(os.getcwd())
    model_path = os.path.join(rootdir , "model")

    img_dir = os.path.join(rootdir, "Dataset", "All_Cropped")
    test_csv = pd.read_csv(os.path.join(rootdir, "Dataset", "test.csv"))

    transform_dict = {
        'jitter':  transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]),
        'h_flip': transforms.RandomHorizontalFlip(),
        'v_flip': transforms.RandomVerticalFlip(),
        'rotate_90': transforms.RandomApply([transforms.RandomRotation(degrees=90)]),
        'rotate_180': transforms.RandomApply([transforms.RandomRotation(degrees=180)]),
        'rotate_270': transforms.RandomApply([transforms.RandomRotation(degrees=270)])
    }
    transform_dict['all'] = transforms.Compose([transform_dict[key] for key in transform_dict])
    transform_dict['normal'] = transforms.RandomHorizontalFlip(p=0)
    
    scores = []
    print("Parameter Grid: ", list(ParameterGrid(parameter_grid)))
    for parameters in ParameterGrid(parameter_grid):
        if parameters["Models"] == "vit_h_14":
            image_size = 518
            batch_size = 4
            learnig_rate = 0.01
        elif parameters["Models"] == "resnext101_32x8d":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = 0.0001
        elif parameters["Models"] == "wide_resnet101_2":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = 0.00001
        elif parameters["Models"] == "convnext_large":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = parameters["Learning Rate"]
        elif parameters["Models"] == "efficientnet_v2_l":
            image_size = 224
            batch_size = 32
            learnig_rate = 0.00001
        elif parameters["Models"] == "regnet_y_128gf":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = parameters["Learning Rate"]
        elif parameters["Models"] == "vgg19":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = parameters["Learning Rate"]
        elif parameters["Models"] == "vgg19_bn":
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = parameters["Learning Rate"]
        else:
            image_size = 224
            batch_size = parameters["Batch Size"]
            learnig_rate = parameters["Learning Rate"]
        print(parameters)

        ssl.get_default_verify_paths()

        model = get_model(parameters["Models"], parameters["Dropout"])
        params = {"Learning Rate": learnig_rate, "Batch Size": batch_size,
                    "Dropout" : parameters["Dropout"], 
                    "Data Augmentation": data_aug_key, "Model": parameters["Models"],
                    "Weighted Random Sampler": True,
                    "New Distribuition": True}
            
        train_fold = pd.read_csv(f'/home/ecarvalho/Rectal_Cancer/Dataset/train.csv')
        if parameters["Weighted Random Sampler"]:
            class_count = train_fold.Label.value_counts()
            print("--------------------------class_count: ", class_count)
            class_weights = 1 /class_count
            train_samples_weight = np.array([class_weights[t] for t in (train_fold.Label.values)])
            train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight), replacement=True)
        else:
            train_sampler = None

        val_fold = pd.read_csv(f'/home/ecarvalho/Rectal_Cancer/Dataset/val.csv')

        mean, std = utils.get_mean_std(train_fold, img_dir)
        params[f"Train Imgs Mean"] = mean; params[f"Train Imgs Std"] = std

        params[f"Train Size"] = len(train_fold); params[f"Val Size"] = len(val_fold); params[f"Test Size"] = len(test_csv)
        params[f"Train Imgs Mean"] = mean; params[f"Train Imgs Std"] = std
                        
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
                    transform_dict[data_aug_key],
                    transforms.ToTensor(), #])
                        transforms.Normalize(mean,std)]) 
        
        transform_val_test = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), #])
                transforms.Normalize(mean,std)])  
              
        comet_logger = CometLogger(
                        api_key="ksdT0SIoCUBmhmzpmTs9uDKCm",
                        project="rectal_3-classes",
                        workspace="eduardo-carvalho"
                        )            
            
        for n in range(n_classes):
            params[f"Train Label {n}"] = len(train_fold[train_fold["Label"]==n])
            params[f"Val Label {n}"] = len(val_fold[val_fold["Label"]==n])
            params[f"Test Label {n}"] = len(test_csv[test_csv["Label"]==n])   
        comet_logger.log_hyperparams(params)                
        
        experiment_id = comet_logger.experiment.id
        model_name = str(experiment_id)

        train_dataset = utils.Dataset(train_fold, img_dir, transform_train, mean, std)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, generator=g, persistent_workers=True, num_workers = 100, drop_last=True, sampler=train_sampler)
                                                            
        val_dataset = utils.Dataset(val_fold, img_dir, transform_val_test, mean, std)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, pin_memory=True, generator=g, persistent_workers=True, num_workers = 100,)
            
        test_dataset = utils.Dataset(test_csv, img_dir, transform_val_test, mean, std)
        test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle = False, pin_memory=True, generator=g, num_workers = 100)
            
        model_pl = utils.ModelPatients(model, float(2), learnig_rate, device)

        early_stop_callback = EarlyStopping(monitor = "Val Accuracy", min_delta=0.00, patience = pat, verbose = False, mode = "max")
        save_model = ModelCheckpoint(dirpath=model_path , filename= model_name, mode="max", monitor="Val Accuracy")
        trainer = Trainer(max_epochs=100, logger=comet_logger, callbacks= [early_stop_callback, save_model], accelerator="gpu", devices=1, precision="16-mixed")

        trainer.fit(model_pl, train_dataloader, val_dataloader)   
        
        trainer.test(model= model_pl, dataloaders= test_dataloader, ckpt_path=os.path.join(model_path, model_name + ".ckpt"))
        comet_logger.experiment.end()