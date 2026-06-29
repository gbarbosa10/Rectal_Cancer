import pytorch_lightning as pl
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import AUROC
from torchmetrics.classification import Accuracy
from torchmetrics.classification import Recall
from torchmetrics.classification import Precision
from torchmetrics.classification import F1Score
from torchmetrics.classification import FBetaScore
from torchmetrics.classification import ROC
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MatthewsCorrCoef
from neptune.types import File
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
#from autoattack import AutoAttack
import sys
import torch.distributed as dist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/ecarvalho/Rectal_Cancer/utils')

class ModelPatients(pl.LightningModule):
    def __init__(self, model, beta, learning_rate, device_): 
        super().__init__()  
        self.model = model    
        self.beta = beta
        self.learning_rate = learning_rate
        self.auroc = AUROC(task="multiclass", num_classes=3)
        self.roc = ROC(task="multiclass", num_classes=3)
        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.acc_class = Accuracy(task="multiclass", num_classes=3)
        self.recall = Recall(task="multiclass", num_classes=3)
        self.recall_class = Recall(task="multiclass", num_classes=3)
        self.precision = Precision(task="multiclass", num_classes=3)
        self.precision_class = Precision(task="multiclass", num_classes=3)
        self.f1score = F1Score(task="multiclass", num_classes=3)
        self.f1score_class = F1Score(task="multiclass", num_classes=3)
        self.fbetascore = FBetaScore(task="multiclass", num_classes=3, beta=self.beta)
        self.fbetascore_class = FBetaScore(task="multiclass", num_classes=3, beta=self.beta)
        self.cm = ConfusionMatrix(task="multiclass", num_classes=3)
        self.MCC = MatthewsCorrCoef(task="multiclass", num_classes=3)
        self.device_name = device_

        self.loss = nn.CrossEntropyLoss().to(self.device_name)
        self.misclassified_imgs=[]
        # self.model_feat.requires_grad_(False) 
        
    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        image, label, img_path = batch

        logits = self.forward(image)
        probs = torch.softmax(logits, dim=1)

        loss = self.loss(logits, label.long())

        preds = torch.argmax(probs, dim=1)

        self.log("Train Accuracy", self.acc(preds, label), on_epoch=True)
        self.log("Train Loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, label, img_path = batch

        logits = self.forward(image)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        loss = self.loss(logits, label.long())

        self.log("Val Accuracy", self.acc(preds, label), on_epoch=True)
        self.log("Val Loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        image, label, img_path = batch

        logits = self.forward(image)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        loss = self.loss(logits, label.long())

        # ✅ Correct metrics
        self.log("Test Accuracy", self.acc(preds, label), on_epoch=True)
        self.log("Test Precision", self.precision(preds, label), on_epoch=True)
        self.log("Test Recall", self.recall(preds, label), on_epoch=True)
        self.log("Test F1", self.f1score(preds, label), on_epoch=True)
        self.log("Test Fb Score", self.fbetascore(preds, label), on_epoch=True)

        # ✅ AUROC uses probabilities
        self.log("Test AUC", self.auroc(probs, label), on_epoch=True)

        self.log("Test Loss", loss, on_epoch=True)

        # ✅ ROC + Conf Matrix
        self.roc.update(probs, label)
        self.cm.update(preds, label)

        # ✅ Misclassified
        misclassified_idx = (preds != label).nonzero(as_tuple=True)[0]

        for idx in misclassified_idx:
            self.misclassified_imgs.append(img_path[idx.item()])

    def on_test_epoch_end(self):

        # ✅ Save misclassified
        df = pd.DataFrame(self.misclassified_imgs)
        df.to_csv("misclassified.csv", index=False)

        # ✅ ROC Curve
        fig_roc, ax_roc = self.roc.plot(score=True)

        self.logger.experiment.log_figure(
            figure=fig_roc,
            figure_name="Test ROC Curve"
        )

        plt.close(fig_roc)

        # ✅ Confusion Matrix (FIX: 3 classes)
        fig_cm, ax_cm = self.cm.plot(labels=[str(i) for i in range(3)])

        self.logger.experiment.log_figure(
            figure=fig_cm,
            figure_name="Test Confusion Matrix"
        )

        plt.close(fig_cm)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)