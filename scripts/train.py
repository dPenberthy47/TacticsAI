#!/usr/bin/env python3
"""
TacticsAI Training Script
Train the GNN model on football tactical matchups.

Usage:
    python scripts/train.py --epochs 10 --max_matches 100
    python scripts/train.py --batch_size 16 --lr 0.0005
"""

import argparse
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path.parent))

import torch
from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "backend/logs/training_{time}.log",
    rotation="10 MB",
    level="DEBUG",
)

from data.dataset import MatchupDataset
from models.gnn import TacticsGNN, SimpleTacticsGNN
from models.trainer import (
    TacticsTrainer,
    TrainerConfig,
    create_paired_dataloader,
    get_device,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TacticsAI GNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    
    # Data parameters
    parser.add_argument(
        "--data_source",
        type=str,
        default="multi",
        choices=["multi", "statsbomb", "synthetic"],
        help="Data source: 'multi' for all sources, 'statsbomb' for StatsBomb only, 'synthetic' for testing",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2015-2016",
        help="Season for multi-source data (e.g., '2015-2016')",
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=None,
        help="Maximum number of matches to load (None = all available)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (deprecated, use --data_source=synthetic)",
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    
    # Model parameters
    parser.add_argument(
        "--node_features",
        type=int,
        default=12,
        help="Number of node features (8 for legacy, 12 for enhanced)",
    )
    parser.add_argument(
        "--edge_features",
        type=int,
        default=6,
        help="Number of edge features (3 for legacy, 6 for enhanced)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GAT layers",
    )
    parser.add_argument(
        "--simple_model",
        action="store_true",
        help="Use simplified model for faster training",
    )
    
    # Output parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="backend/models/saved/tacticsai.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save the model",
    )
    
    return parser.parse_args()


def validate_dataset(dataset) -> None:
    """Validate dataset has proper enhanced features."""
    logger.info("=" * 60)
    logger.info("Validating dataset...")

    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return

    try:
        sample = dataset[0]
        graph_a = sample[0] if isinstance(sample, tuple) else sample.get('graph_a')

        node_dim = graph_a.x.shape[1]
        edge_dim = graph_a.edge_attr.shape[1] if graph_a.edge_attr is not None else 0

        logger.info("Dataset validation:")
        logger.info(f"  Samples: {len(dataset)}")
        logger.info(f"  Node features: {node_dim} (expected: 12)")
        logger.info(f"  Edge features: {edge_dim} (expected: 6)")
        logger.info(f"  Nodes per graph: {graph_a.x.shape[0]}")

        if node_dim < 12:
            logger.warning("⚠ Using legacy node features (8D). Enhanced features (12D) recommended.")
        else:
            logger.info("✓ Using enhanced node features (12D)")

        if edge_dim < 6:
            logger.warning("⚠ Using legacy edge features (3D). Enhanced features (6D) recommended.")
        else:
            logger.info("✓ Using enhanced edge features (6D)")

    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")


def load_data(args) -> tuple:
    """Load and split the dataset."""
    logger.info("=" * 60)
    logger.info("Loading data...")

    # Handle legacy --synthetic flag
    if args.synthetic:
        args.data_source = "synthetic"

    # Load based on data source
    if args.data_source == "multi":
        logger.info(f"Loading multi-source data for season {args.season}")
        dataset = MatchupDataset.from_multi_source(
            season=args.season,
            max_matches=args.max_matches,
        )
    elif args.data_source == "statsbomb":
        logger.info("Loading from StatsBomb open data only")
        dataset = MatchupDataset.from_statsbomb(
            competition_id=2,  # Premier League
            season_id=27,      # 2015/2016 (free data)
            max_matches=args.max_matches,
        )
    else:  # synthetic
        logger.info(f"Creating synthetic dataset with {args.synthetic_samples} samples")
        dataset = MatchupDataset.create_synthetic(n_samples=args.synthetic_samples)

    # Validate dataset features
    validate_dataset(dataset)
    
    # Split into train/val
    train_dataset, val_dataset = dataset.split(
        val_ratio=args.val_split,
        shuffle=True,
        seed=42,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = create_paired_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_paired_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def create_model(args) -> torch.nn.Module:
    """Initialize the model."""
    logger.info("=" * 60)
    logger.info("Creating model...")

    if args.simple_model:
        logger.info("Using SimpleTacticsGNN")
        model = SimpleTacticsGNN(
            node_features=args.node_features,
            edge_features=args.edge_features,
            hidden_dim=args.hidden_dim,
            use_edge_features=True,
        )
    else:
        logger.info("Using TacticsGNN")
        model = TacticsGNN(
            node_features=args.node_features,
            edge_features=args.edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_edge_features=True,
        )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model configuration:")
    logger.info(f"  Node features: {args.node_features}D")
    logger.info(f"  Edge features: {args.edge_features}D")
    logger.info(f"  Hidden dimension: {args.hidden_dim}")
    logger.info(f"  Layers: {args.num_layers if not args.simple_model else 2}")
    logger.info(f"  Parameters: {num_params:,}")

    return model


def train_model(args, model, train_loader, val_loader):
    """Run the training loop."""
    logger.info("=" * 60)
    logger.info("Starting training...")
    
    # Create trainer config
    config = TrainerConfig(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=str(Path(args.model_path).parent),
    )
    
    # Initialize trainer
    trainer = TacticsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train
    history = trainer.train()
    
    # Save final model
    if not args.no_save:
        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_path)
        logger.info(f"Model saved to {model_path}")
    
    return trainer, history


def evaluate_model(trainer, val_dataset, n_samples: int = 5):
    """Run evaluation and print sample predictions."""
    logger.info("=" * 60)
    logger.info("Evaluation")
    
    # Load best model
    best_path = Path(trainer.config.save_dir) / "best_model.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)
        logger.info("Loaded best model for evaluation")
    
    # Final validation metrics
    val_metrics = trainer.validate()
    
    logger.info("-" * 40)
    logger.info("Final Metrics:")
    logger.info(f"  Validation Loss: {val_metrics['val_loss']:.4f}")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.2%}")
    logger.info(f"  MAE: {val_metrics['mae']:.4f}")
    
    # Sample predictions
    logger.info("-" * 40)
    logger.info("Sample Predictions:")
    
    device = get_device()
    trainer.model.eval()
    
    for i in range(min(n_samples, len(val_dataset))):
        try:
            sample = val_dataset.get(i)
            
            if len(sample) == 3:
                graph_a, graph_b, label = sample
                if isinstance(label, tuple):
                    label = label[0]
            else:
                continue
            
            graph_a = graph_a.to(device)
            graph_b = graph_b.to(device)
            
            with torch.no_grad():
                output = trainer.model(graph_a, graph_b)
                pred = output["dominance"].item()
            
            true_label = label.item() if isinstance(label, torch.Tensor) else label
            
            # Interpret prediction
            if pred > 0.6:
                pred_text = "Team A dominates"
            elif pred < 0.4:
                pred_text = "Team B dominates"
            else:
                pred_text = "Balanced"
            
            if true_label > 0.6:
                true_text = "Team A won"
            elif true_label < 0.4:
                true_text = "Team B won"
            else:
                true_text = "Draw"
            
            correct = "✓" if (pred > 0.5) == (true_label > 0.5) else "✗"
            
            logger.info(
                f"  Sample {i + 1}: {correct} "
                f"Pred={pred:.3f} ({pred_text}) | "
                f"True={true_label:.1f} ({true_text})"
            )
            
        except Exception as e:
            logger.debug(f"Error on sample {i}: {e}")
            continue
    
    return val_metrics


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("TacticsAI Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    
    # Ensure logs directory exists
    Path("backend/logs").mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        train_loader, val_loader, train_dataset, val_dataset = load_data(args)
        
        # Create model
        model = create_model(args)
        
        # Train
        trainer, history = train_model(args, model, train_loader, val_loader)
        
        # Evaluate
        val_metrics = evaluate_model(trainer, val_dataset)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

