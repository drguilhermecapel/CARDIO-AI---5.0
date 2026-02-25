# Trainer clínico-grade
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy

from src.data.augmentation import MixupCutmix

class ModelEMA:
    """Exponential Moving Average dos pesos do modelo"""
    def __init__(self, model, decay=0.9999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd:
                    v.copy_(self.decay * v + (1. - self.decay) * msd[k])

class ClinicalECGTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.training_config.learning_rate,
            weight_decay=config.training_config.weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.training_config.warmup_epochs * 2,
            T_mult=2
        )
        self.logger = logging.getLogger(__name__)
        
        # Advanced features
        self.ema = ModelEMA(model, decay=config.training_config.ema_decay) if config.training_config.use_ema else None
        self.mixup_fn = MixupCutmix(config) if (config.training_config.use_mixup or config.training_config.use_cutmix) else None
        self.scaler = torch.cuda.amp.GradScaler() if config.training_config.mixed_precision else None
        
    def train_epoch(self, train_loader):
        """Treinamento de um epoch com Mixup/Cutmix e Gradient Accumulation"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Mixup/Cutmix
            if self.mixup_fn:
                batch_x, y_a, y_b, lam = self.mixup_fn(batch_x, batch_y)
            
            # Mixed Precision
            with torch.cuda.amp.autocast(enabled=self.config.training_config.mixed_precision):
                output = self.model(batch_x)
                
                if self.mixup_fn:
                    loss = lam * self.compute_loss(output.logits, y_a) + \
                           (1 - lam) * self.compute_loss(output.logits, y_b)
                else:
                    loss = self.compute_loss(output.logits, batch_y)
                
                loss = loss / self.config.training_config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient Accumulation Step
            if (i + 1) % self.config.training_config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema:
                    self.ema.update(self.model)
            
            total_loss += loss.item() * self.config.training_config.gradient_accumulation_steps
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validação usando modelo EMA se disponível"""
        model_to_eval = self.ema.module if self.ema else self.model
        model_to_eval.eval()
        
        val_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = model_to_eval(batch_x)
            loss = self.compute_loss(output.logits, batch_y)
            val_loss += loss.item()
            
            all_preds.append(output.probabilities.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calcular métricas detalhadas
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = self.compute_detailed_metrics(all_preds, all_targets)
        
        return val_loss, metrics
    
    def compute_loss(self, logits, targets):
        """Loss function ponderada com label smoothing e pesos específicos para arritmias ventriculares"""
        if self.config.training_config.label_smoothing > 0:
            targets = targets * (1 - self.config.training_config.label_smoothing) + \
                      0.5 * self.config.training_config.label_smoothing
                      
        # Construir tensor de pesos baseado na configuração
        num_classes = logits.shape[-1]
        pos_weights = torch.ones(num_classes, device=logits.device) * 2.0  # Peso base
        
        # Aplicar pesos específicos se configurados
        if hasattr(self.config.training_config, 'class_weights') and self.config.training_config.class_weights:
            from src.config.config import DiagnosticClass
            for class_name, weight in self.config.training_config.class_weights.items():
                try:
                    # Encontrar o índice da classe no enum
                    class_idx = list(DiagnosticClass).index(DiagnosticClass[class_name])
                    if class_idx < num_classes:
                        pos_weights[class_idx] = weight
                except (KeyError, ValueError):
                    pass
                      
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=pos_weights
        )
    
    def compute_detailed_metrics(self, preds, targets):
        """Calcula AUROC macro e por classe"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        try:
            auroc_macro = roc_auc_score(targets, preds, average='macro')
            auprc_macro = average_precision_score(targets, preds, average='macro')
        except ValueError:
            auroc_macro = 0.0
            auprc_macro = 0.0
            
        # Per-class metrics (para log)
        class_aurocs = {}
        for i in range(targets.shape[1]):
            try:
                score = roc_auc_score(targets[:, i], preds[:, i])
                class_aurocs[f"class_{i}"] = score
            except ValueError:
                pass
                
        return {
            "auroc_macro": auroc_macro,
            "auprc_macro": auprc_macro,
            "class_metrics": class_aurocs
        }
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Loop de treinamento completo com early stopping"""
        best_metric = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            current_metric = val_metrics["auroc_macro"]
            
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"AUROC={val_metrics['auroc_macro']:.4f}, "
                f"AUPRC={val_metrics['auprc_macro']:.4f}"
            )
            
            # Log per-class metrics for critical classes
            # (Exemplo: Class 0 = AFib, Class 1 = STEMI)
            # self.logger.info(f"  AFib AUROC: {val_metrics['class_metrics'].get('class_0', 0):.4f}")
            
            if current_metric > best_metric + self.config.training_config.early_stopping_min_delta:
                best_metric = current_metric
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                if self.ema:
                    torch.save(self.ema.module.state_dict(), 'best_model_ema.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.training_config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
            
            self.scheduler.step()
        
        return best_metric
