"""
TacticsAI Model Trainer
Training infrastructure for the GNN-based tactics prediction model.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not installed. Graph batching will be limited.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not installed. Progress bars will be disabled.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for the TacticsTrainer."""
    
    # Optimization
    lr: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 50
    patience: int = 10
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Loss weights
    dominance_weight: float = 1.0
    confidence_weight: float = 0.5
    battle_zones_weight: float = 0.3
    
    # Scheduler
    scheduler_t_max: Optional[int] = None  # Defaults to epochs if None
    scheduler_eta_min: float = 1e-6
    
    # Checkpointing
    save_dir: str = "backend/models/saved"
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 10  # Log every N batches


# =============================================================================
# Custom Collate Function for Graph Pairs
# =============================================================================

def collate_graph_pairs(batch: List[Tuple[Data, Data, torch.Tensor]]) -> Tuple[Batch, Batch, torch.Tensor]:
    """
    Custom collate function for paired graph data.
    
    Batches graph_a's together, graph_b's together, and stacks labels.
    
    Args:
        batch: List of (graph_a, graph_b, label) tuples
        
    Returns:
        Tuple of (batched_graph_a, batched_graph_b, stacked_labels)
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric required for graph batching")
    
    graphs_a = []
    graphs_b = []
    labels = []
    
    for graph_a, graph_b, label in batch:
        graphs_a.append(graph_a)
        graphs_b.append(graph_b)
        labels.append(label)
    
    # Batch graphs using PyG's Batch class
    batched_a = Batch.from_data_list(graphs_a)
    batched_b = Batch.from_data_list(graphs_b)
    
    # Stack labels into a tensor
    if isinstance(labels[0], torch.Tensor):
        stacked_labels = torch.stack(labels)
    else:
        stacked_labels = torch.tensor(labels)
    
    return batched_a, batched_b, stacked_labels


def create_paired_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for paired graph data.
    
    Args:
        dataset: Dataset yielding (graph_a, graph_b, label) tuples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with custom collate function
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_pairs,
        **kwargs
    )


# =============================================================================
# Device Detection
# =============================================================================

def get_device() -> torch.device:
    """
    Auto-detect the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device for the best available hardware
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


# =============================================================================
# Tactics Trainer
# =============================================================================

class TacticsTrainer:
    """
    Trainer for the GNN-based football tactics prediction model.
    
    Handles training loop, validation, early stopping, checkpointing,
    and multi-task loss computation for:
    - Dominance prediction (regression)
    - Confidence estimation (binary classification)
    - Battle zones prediction (multi-label classification)
    
    Usage:
        model = TacticsGNN(...)
        trainer = TacticsTrainer(model, train_loader, val_loader)
        history = trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainerConfig] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration (uses defaults if None)
        """
        self.config = config or TrainerConfig()
        self.device = get_device()
        
        # Model
        self.model = model.to(self.device)
        logger.info(f"Model has {self._count_parameters():,} trainable parameters")
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        t_max = self.config.scheduler_t_max or self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=self.config.scheduler_eta_min,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_dominance_loss": [],
            "train_confidence_loss": [],
            "train_battle_zones_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_mae": [],
            "lr": [],
        }
        
        # Ensure save directory exists
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TacticsTrainer initialized | device={self.device} | epochs={self.config.epochs}")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    # -------------------------------------------------------------------------
    # Loss Functions
    # -------------------------------------------------------------------------
    
    def dominance_loss(
        self,
        pred_dominance: torch.Tensor,
        true_dominance: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss for dominance prediction.
        
        Dominance is a continuous value representing tactical advantage.
        
        Args:
            pred_dominance: Predicted dominance scores
            true_dominance: Ground truth dominance scores
            
        Returns:
            MSE loss tensor
        """
        return F.mse_loss(pred_dominance, true_dominance)
    
    def confidence_loss(
        self,
        pred_confidence: torch.Tensor,
        true_dominance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross entropy for confidence estimation.
        
        High confidence should correlate with clear dominance (far from 0.5).
        Label certainty = |dominance - 0.5| * 2 (scaled to [0, 1])
        
        Args:
            pred_confidence: Predicted confidence (0-1)
            true_dominance: Ground truth dominance scores
            
        Returns:
            BCE loss tensor
        """
        # Calculate label certainty from dominance
        # Dominance of 0 or 1 = high certainty, dominance of 0.5 = low certainty
        label_certainty = torch.abs(true_dominance - 0.5) * 2
        label_certainty = label_certainty.clamp(0, 1)
        
        return F.binary_cross_entropy(
            pred_confidence.clamp(1e-7, 1 - 1e-7),
            label_certainty,
        )
    
    def battle_zones_loss(
        self,
        pred_zones: torch.Tensor,
        true_zones: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss for battle zones prediction.

        Battle zones are now attention-derived continuous values (0-1)
        representing relative tactical control per zone.
        Changed from BCE to MSE to handle continuous targets.

        Args:
            pred_zones: Predicted zone values (0-1, attention-derived)
            true_zones: Ground truth zone values (0-1, continuous)

        Returns:
            MSE loss tensor
        """
        # Battle zones are now continuous (0-1), not binary
        # Use MSE instead of BCE
        return F.mse_loss(pred_zones, true_zones)
    
    def compute_total_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted total loss from all components.

        Multi-task loss for:
        - Dominance: MSE loss (continuous 0-1 from tactical dominance calculation)
        - Confidence: BCE loss (certainty of prediction)
        - Battle zones: MSE loss (attention-derived continuous zone control)

        Args:
            predictions: Dict with 'dominance', 'confidence', 'battle_zones'
            targets: Dict with 'dominance', 'battle_zones'

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Dominance loss - regression on continuous target (0-1)
        dom_loss = self.dominance_loss(
            predictions["dominance"],
            targets["dominance"],
        )

        # Confidence loss (optional)
        conf_loss = torch.tensor(0.0, device=self.device)
        if "confidence" in predictions:
            conf_loss = self.confidence_loss(
                predictions["confidence"],
                targets["dominance"],
            )

        # Battle zones loss (optional) - now MSE for continuous targets
        zones_loss = torch.tensor(0.0, device=self.device)
        if "battle_zones" in predictions and "battle_zones" in targets:
            zones_loss = self.battle_zones_loss(
                predictions["battle_zones"],
                targets["battle_zones"],
            )

        # Weighted total
        total = (
            self.config.dominance_weight * dom_loss +
            self.config.confidence_weight * conf_loss +
            self.config.battle_zones_weight * zones_loss
        )

        loss_components = {
            "dominance": dom_loss.item(),
            "confidence": conf_loss.item(),
            "battle_zones": zones_loss.item(),
            "total": total.item(),
        }

        return total, loss_components
    
    # -------------------------------------------------------------------------
    # Training Methods
    # -------------------------------------------------------------------------
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with average loss values for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_dominance_loss = 0.0
        total_confidence_loss = 0.0
        total_zones_loss = 0.0
        num_batches = 0
        
        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}",
                leave=False,
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                graph_a, graph_b, labels = batch
                graph_a = graph_a.to(self.device)
                graph_b = graph_b.to(self.device)
                # Handle labels - may be a list with mixed shapes (dominance + battle_zones)
                if isinstance(labels, list):
                    # If it's a list of same-shaped tensors, stack them
                    # If mixed shapes, just use the first element (dominance labels)
                    try:
                        labels = torch.stack(labels).to(self.device)
                    except RuntimeError:
                        # Mixed shapes - extract just dominance labels (1D tensors)
                        labels = torch.cat([l.view(-1) for l in labels if l.dim() == 1 or l.numel() == l.shape[0]]).to(self.device)
                else:
                    labels = labels[0].to(self.device) if isinstance(labels, list) else labels.to(self.device)
            else:
                # Single graph batch (for simpler models)
                batch = batch.to(self.device)
                graph_a = batch
                graph_b = None
                labels = batch.y if hasattr(batch, "y") else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if graph_b is not None:
                outputs = self.model(graph_a, graph_b)
            else:
                outputs = self.model(graph_a)
            
            # Prepare targets
            targets = self._prepare_targets(labels)
            
            # Prepare predictions dict
            predictions = self._prepare_predictions(outputs)
            
            # Compute loss
            loss, loss_components = self.compute_total_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_components["total"]
            total_dominance_loss += loss_components["dominance"]
            total_confidence_loss += loss_components["confidence"]
            total_zones_loss += loss_components["battle_zones"]
            num_batches += 1
            
            # Update progress bar
            if TQDM_AVAILABLE and batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss_components['total']:.4f}",
                    "dom": f"{loss_components['dominance']:.4f}",
                })
        
        # Compute averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_dominance = total_dominance_loss / max(num_batches, 1)
        avg_confidence = total_confidence_loss / max(num_batches, 1)
        avg_zones = total_zones_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "dominance_loss": avg_dominance,
            "confidence_loss": avg_confidence,
            "battle_zones_loss": avg_zones,
        }
    
    def _prepare_targets(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare target tensors from labels.
        
        Handles different label formats (scalar, tuple, dict).
        """
        if isinstance(labels, dict):
            return {k: v.to(self.device) for k, v in labels.items()}
        
        if isinstance(labels, (tuple, list)):
            # Assume (dominance, battle_zones) format
            targets = {"dominance": labels[0].to(self.device)}
            if len(labels) > 1:
                targets["battle_zones"] = labels[1].to(self.device)
            return targets
        
        # Single tensor = dominance score
        return {"dominance": labels.float().to(self.device)}
    
    def _prepare_predictions(self, outputs) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions dict from model outputs.
        
        Handles different output formats.
        """
        if isinstance(outputs, dict):
            return outputs
        
        if isinstance(outputs, (tuple, list)):
            predictions = {"dominance": outputs[0]}
            if len(outputs) > 1:
                predictions["confidence"] = outputs[1]
            if len(outputs) > 2:
                predictions["battle_zones"] = outputs[2]
            return predictions
        
        # Single tensor = dominance score
        return {"dominance": outputs}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation without computing gradients.
        
        Returns:
            Dict with val_loss, accuracy, and MAE
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        correct_direction = 0
        total_samples = 0
        
        for batch in self.val_loader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                graph_a, graph_b, labels = batch
                graph_a = graph_a.to(self.device)
                graph_b = graph_b.to(self.device)
                labels = labels[0].to(self.device) if isinstance(labels, list) else labels.to(self.device)
            else:
                batch = batch.to(self.device)
                graph_a = batch
                graph_b = None
                labels = batch.y if hasattr(batch, "y") else None
            
            # Forward pass
            if graph_b is not None:
                outputs = self.model(graph_a, graph_b)
            else:
                outputs = self.model(graph_a)
            
            # Prepare targets and predictions
            targets = self._prepare_targets(labels)
            predictions = self._prepare_predictions(outputs)
            
            # Compute loss
            loss, _ = self.compute_total_loss(predictions, targets)
            total_loss += loss.item()
            
            # Compute MAE for dominance
            pred_dominance = predictions["dominance"]
            true_dominance = targets["dominance"]
            
            total_mae += torch.abs(pred_dominance - true_dominance).sum().item()
            
            # Compute directional accuracy
            # Correct if both predict same team dominance (>0.5 or <0.5)
            pred_direction = (pred_dominance > 0.5).float()
            true_direction = (true_dominance > 0.5).float()
            correct_direction += (pred_direction == true_direction).sum().item()
            
            total_samples += pred_dominance.numel()
        
        num_batches = len(self.val_loader)
        
        return {
            "val_loss": total_loss / max(num_batches, 1),
            "accuracy": correct_direction / max(total_samples, 1),
            "mae": total_mae / max(total_samples, 1),
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history dict
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Patience: {self.config.patience}")
        logger.info(f"  Learning rate: {self.config.lr}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_dominance_loss"].append(train_metrics["dominance_loss"])
            self.history["train_confidence_loss"].append(train_metrics["confidence_loss"])
            self.history["train_battle_zones_loss"].append(train_metrics["battle_zones_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["lr"].append(current_lr)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.2%} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Check for improvement
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                
                # Save best model
                if self.config.save_best_only:
                    best_path = self.save_dir / "best_model.pt"
                    self.save_checkpoint(best_path)
                    logger.info(f"  ✓ New best model saved (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                logger.debug(f"  No improvement. Patience: {self.patience_counter}/{self.config.patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Total time: {total_time / 60:.1f} minutes")
        logger.info(f"  Best val_loss: {self.best_val_loss:.4f}")
        logger.info(f"  Final accuracy: {self.history['val_accuracy'][-1]:.2%}")
        logger.info("=" * 60)
        
        # Save final model
        final_path = self.save_dir / "final_model.pt"
        self.save_checkpoint(final_path)
        
        return self.history
    
    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    
    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "config": {
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "epochs": self.config.epochs,
                "patience": self.config.patience,
                "max_grad_norm": self.config.max_grad_norm,
            },
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        self.history = checkpoint.get("history", self.history)
        
        logger.info(f"Checkpoint loaded from {path} (epoch {self.current_epoch})")
    
    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    
    @torch.no_grad()
    def predict(
        self,
        graph_a: Data,
        graph_b: Optional[Data] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction for a single graph pair.
        
        Args:
            graph_a: First team's tactical graph
            graph_b: Second team's tactical graph (optional)
            
        Returns:
            Dict with predictions
        """
        self.model.eval()
        
        graph_a = graph_a.to(self.device)
        if graph_b is not None:
            graph_b = graph_b.to(self.device)
            outputs = self.model(graph_a, graph_b)
        else:
            outputs = self.model(graph_a)
        
        return self._prepare_predictions(outputs)

