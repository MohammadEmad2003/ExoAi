"""
Background worker for heavy ML training jobs using asyncio queues.
This provides a simple alternative to Celery for handling training tasks.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import uuid

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingWorker:
    """Async worker for handling training jobs"""
    
    def __init__(self, max_concurrent_jobs: int = 2):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = asyncio.Queue()
        self.active_jobs: Dict[str, Dict] = {}
        self.completed_jobs: Dict[str, Dict] = {}
        self.is_running = False
        
    async def start(self):
        """Start the worker"""
        self.is_running = True
        logger.info(f"Starting training worker with {self.max_concurrent_jobs} concurrent jobs")
        
        # Start worker tasks
        tasks = []
        for i in range(self.max_concurrent_jobs):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the worker"""
        self.is_running = False
        logger.info("Stopping training worker")
    
    async def submit_job(self, job_data: Dict[str, Any]) -> str:
        """Submit a training job to the queue"""
        job_id = str(uuid.uuid4())
        job_data['job_id'] = job_id
        job_data['status'] = 'queued'
        job_data['submitted_at'] = datetime.now().isoformat()
        
        await self.job_queue.put(job_data)
        logger.info(f"Job {job_id} submitted to queue")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    def get_all_jobs(self) -> Dict[str, Dict]:
        """Get all jobs (active and completed)"""
        all_jobs = {}
        all_jobs.update(self.active_jobs)
        all_jobs.update(self.completed_jobs)
        return all_jobs
    
    async def _worker_loop(self, worker_name: str):
        """Main worker loop"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get job from queue (with timeout to allow checking is_running)
                job_data = await asyncio.wait_for(
                    self.job_queue.get(), 
                    timeout=1.0
                )
                
                job_id = job_data['job_id']
                logger.info(f"Worker {worker_name} processing job {job_id}")
                
                # Move job to active
                self.active_jobs[job_id] = job_data
                
                # Process the job
                await self._process_training_job(job_data, worker_name)
                
                # Move job to completed
                self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                # No job available, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                if job_id in self.active_jobs:
                    self.active_jobs[job_id]['status'] = 'failed'
                    self.active_jobs[job_id]['error'] = str(e)
                    self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
    
    async def _process_training_job(self, job_data: Dict, worker_name: str):
        """Process a single training job"""
        job_id = job_data['job_id']
        
        try:
            # Update job status
            job_data['status'] = 'running'
            job_data['started_at'] = datetime.now().isoformat()
            job_data['worker'] = worker_name
            job_data['progress'] = 0.0
            
            # Extract job parameters
            file_path = job_data['file_path']
            target_column = job_data['target_column']
            config = job_data['config']
            
            logger.info(f"Training {config['model_type']} on {file_path}")
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Handle target variable
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
                n_classes = len(target_encoder.classes_)
            else:
                target_encoder = None
                n_classes = len(y.unique())
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=config['validation_split'], random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_train_tensor = torch.LongTensor(y_train)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Import model class dynamically
            if config['model_type'] == 'tabkanet':
                from models.tabkanet import TabKANet
                
                model = TabKANet(
                    n_num=X_train.shape[1],
                    n_cat=0,  # Simplified - all encoded as numerical
                    cat_card_list=[],
                    d_model=config['d_model'],
                    K_inner=config['K_inner'],
                    trans_heads=config['trans_heads'],
                    trans_depth=config['trans_depth'],
                    mlp_hidden=config['mlp_hidden'],
                    n_classes=n_classes,
                    dropout=config['dropout']
                )
            else:
                raise ValueError(f"Unsupported model type: {config['model_type']}")
            
            # Training setup
            criterion = torch.nn.CrossEntropyLoss()
            
            if config['optimizer'] == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            elif config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['learning_rate']
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            
            # Learning rate scheduler
            if config['scheduler'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs']
                )
            elif config['scheduler'] == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=config['epochs']//3, gamma=0.1
                )
            else:
                scheduler = None
            
            # Training loop
            model.train()
            best_val_acc = 0.0
            patience_counter = 0
            training_history = []
            
            for epoch in range(config['epochs']):
                # Training step
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_train_tensor, None)
                loss = criterion(outputs, y_train_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                # Validation every few epochs
                if epoch % 5 == 0 or epoch == config['epochs'] - 1:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor, None)
                        val_loss = criterion(val_outputs, y_val_tensor)
                        val_preds = torch.argmax(val_outputs, dim=1)
                        val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())
                    
                    model.train()
                    
                    # Update progress
                    progress = (epoch + 1) / config['epochs'] * 100
                    job_data['progress'] = progress
                    job_data['current_epoch'] = epoch + 1
                    job_data['metrics'] = {
                        'train_loss': float(loss.item()),
                        'val_loss': float(val_loss.item()),
                        'val_accuracy': float(val_acc),
                        'learning_rate': float(optimizer.param_groups[0]['lr'])
                    }
                    
                    training_history.append({
                        'epoch': epoch + 1,
                        'train_loss': float(loss.item()),
                        'val_loss': float(val_loss.item()),
                        'val_accuracy': float(val_acc)
                    })
                    
                    logger.info(f"Job {job_id} - Epoch {epoch+1}/{config['epochs']}: "
                              f"Loss={loss.item():.4f}, Val_Acc={val_acc:.4f}")
                    
                    # Early stopping
                    if config['early_stopping']:
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= config['patience']:
                                logger.info(f"Job {job_id} - Early stopping at epoch {epoch+1}")
                                break
                
                # Small delay to prevent blocking
                if epoch % 10 == 0:
                    await asyncio.sleep(0.01)
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor, None)
                val_preds = torch.argmax(val_outputs, dim=1)
                
                # Calculate comprehensive metrics
                val_acc = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val_tensor.numpy(), val_preds.numpy(), average='weighted'
                )
                
                # Try to calculate AUC if possible
                try:
                    val_probs = torch.softmax(val_outputs, dim=1)
                    if n_classes == 2:
                        auc = roc_auc_score(y_val_tensor.numpy(), val_probs[:, 1].numpy())
                    else:
                        auc = roc_auc_score(
                            y_val_tensor.numpy(), val_probs.numpy(), 
                            multi_class='ovr', average='weighted'
                        )
                except:
                    auc = None
            
            # Save model
            models_dir = Path("models_registry")
            models_dir.mkdir(exist_ok=True)
            
            model_id = str(uuid.uuid4())
            model_path = models_dir / f"{model_id}.pt"
            
            # Save complete model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': config,
                'model_architecture': config['model_type'],
                'scaler': scaler,
                'label_encoders': label_encoders,
                'target_encoder': target_encoder,
                'feature_names': X.columns.tolist(),
                'target_column': target_column,
                'n_classes': n_classes,
                'n_features': X_train.shape[1],
                'training_history': training_history,
                'final_metrics': {
                    'accuracy': float(val_acc),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'auc_roc': float(auc) if auc else None
                }
            }
            
            torch.save(checkpoint, model_path)
            
            # Update job completion
            job_data['status'] = 'completed'
            job_data['progress'] = 100.0
            job_data['completed_at'] = datetime.now().isoformat()
            job_data['model_id'] = model_id
            job_data['model_path'] = str(model_path)
            job_data['final_metrics'] = checkpoint['final_metrics']
            job_data['training_history'] = training_history
            
            logger.info(f"Job {job_id} completed successfully. Model saved as {model_id}")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job_data['status'] = 'failed'
            job_data['error'] = str(e)
            job_data['failed_at'] = datetime.now().isoformat()
            raise

# Global worker instance
training_worker = TrainingWorker(max_concurrent_jobs=2)

async def start_worker():
    """Start the training worker"""
    await training_worker.start()

def get_worker() -> TrainingWorker:
    """Get the global worker instance"""
    return training_worker

if __name__ == "__main__":
    # Run worker standalone
    asyncio.run(start_worker())
