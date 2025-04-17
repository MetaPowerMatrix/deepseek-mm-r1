import os
import json
import logging
import math
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

class QATrainingConfig:
    # 数据配置
    data_dir = "./data"
    train_qa_file = "train/qa_data.jsonl"
    train_reasoning_file = "train/reasoning_data.jsonl"
    val_qa_file = "val/qa_data.jsonl"
    val_reasoning_file = "val/reasoning_data.jsonl"
    max_seq_len = 512
    
    # 训练参数
    batch_size = 8
    gradient_accum_steps = 4
    epochs = 10
    learning_rate = 3e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # 模型配置
    model_name = "heshunyinghua-moe"
    pad_token_id = 0
    eos_token_id = 1
    
    # 日志和保存
    output_dir = "./output"
    log_interval = 50
    save_interval = 1000
    eval_interval = 500
    
    # 分布式
    world_size = torch.cuda.device_count()

class QADataset(Dataset):
    def __init__(self, qa_path, reasoning_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        # 加载QA数据
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.examples.append({
                    'type': 'qa',
                    'question': item['question'],
                    'answer': item['answer']
                })
        
        # 加载推理数据
        with open(reasoning_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.examples.append({
                    'type': 'reasoning',
                    'question': item['question'],
                    'reasoning': item['reasoning'],
                    'answer': item['answer']
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        
        if item['type'] == 'qa':
            # 格式: [CLS]问题[SEP]答案[SEP]
            text = f"{item['question']} {self.tokenizer.sep_token} {item['answer']}"
        else:
            # 格式: [CLS]问题[SEP]推理1[SEP]推理2[SEP]...答案[SEP]
            reasoning = f" {self.tokenizer.sep_token} ".join(item['reasoning'])
            text = f"{item['question']} {self.tokenizer.sep_token} {reasoning} {self.tokenizer.sep_token} {item['answer']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'type': item['type']
        }

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def create_dataloaders(tokenizer, local_rank):
    train_dataset = QADataset(
        os.path.join(QATrainingConfig.data_dir, QATrainingConfig.train_qa_file),
        os.path.join(QATrainingConfig.data_dir, QATrainingConfig.train_reasoning_file),
        tokenizer,
        QATrainingConfig.max_seq_len
    )
    
    val_dataset = QADataset(
        os.path.join(QATrainingConfig.data_dir, QATrainingConfig.val_qa_file),
        os.path.join(QATrainingConfig.data_dir, QATrainingConfig.val_reasoning_file),
        tokenizer,
        QATrainingConfig.max_seq_len
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=QATrainingConfig.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=QATrainingConfig.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

def train_epoch(model, train_loader, optimizer, scheduler, scaler, grad_accum_steps, epoch, global_step, local_rank, logger):
    model.train()
    total_loss = 0.0
    qa_loss = 0.0
    reasoning_loss = 0.0
    qa_count = 0
    reasoning_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=local_rank != 0)
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        batch_types = batch['type']
        
        # 准备labels (shifted input_ids)
        labels = input_ids.clone()
        labels[labels == QATrainingConfig.pad_token_id] = -100
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / grad_accum_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 记录损失
        total_loss += loss.item() * grad_accum_steps
        
        # 分类统计QA和推理的损失
        for i, typ in enumerate(batch_types):
            if typ == 'qa':
                qa_loss += outputs.loss.item()
                qa_count += 1
            else:
                reasoning_loss += outputs.loss.item()
                reasoning_count += 1
        
        # 梯度累积步骤
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), QATrainingConfig.max_grad_norm)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            
            # 日志记录
            if global_step % QATrainingConfig.log_interval == 0 and local_rank == 0:
                avg_qa_loss = qa_loss / max(1, qa_count)
                avg_reasoning_loss = reasoning_loss / max(1, reasoning_count)
                
                logger.info(
                    f"Step {global_step} | "
                    f"Total Loss: {total_loss/(batch_idx+1):.4f} | "
                    f"QA Loss: {avg_qa_loss:.4f} | "
                    f"Reasoning Loss: {avg_reasoning_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            
            # 验证
            if global_step % QATrainingConfig.eval_interval == 0:
                eval_loss = evaluate(model, val_loader, local_rank)
                if local_rank == 0:
                    logger.info(f"Validation Loss: {eval_loss:.4f}")
                model.train()
            
            # 保存检查点
            if global_step % QATrainingConfig.save_interval == 0 and local_rank == 0:
                save_path = os.path.join(QATrainingConfig.output_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }, save_path)
                logger.info(f"Checkpoint saved at step {global_step}")
    
    return global_step, total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, val_loader, local_rank):
    model.eval()
    total_loss = 0.0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = input_ids.clone()
        labels[labels == QATrainingConfig.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 跨GPU聚合损失
        reduced_loss = outputs.loss.detach().clone()
        dist.all_reduce(reduced_loss)
        total_loss += reduced_loss.item() / dist.get_world_size()
    
    return total_loss / len(val_loader)

def main():
    # 初始化分布式训练
    local_rank = setup_distributed()
    
    # 初始化日志
    if local_rank == 0:
        os.makedirs(QATrainingConfig.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(QATrainingConfig.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = None
    
    # 初始化tokenizer (使用适合你模型的tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    # 初始化模型 (替换为你的MoE模型)
    from simple_moe import TransformerMoE
    model = TransformerMoE.from_pretrained(QATrainingConfig.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank])
    
    # 准备数据加载器
    train_loader, val_loader, train_sampler = create_dataloaders(tokenizer, local_rank)
    
    # 计算总训练步数
    total_steps = (len(train_loader) // QATrainingConfig.gradient_accum_steps) * QATrainingConfig.epochs
    warmup_steps = int(total_steps * QATrainingConfig.warmup_ratio)
    
    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=QATrainingConfig.learning_rate,
        weight_decay=QATrainingConfig.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=QATrainingConfig.learning_rate * 0.1
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    global_step = 0
    for epoch in range(QATrainingConfig.epochs):
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            logger.info(f"Starting epoch {epoch + 1}/{QATrainingConfig.epochs}")
        
        global_step, epoch_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            QATrainingConfig.gradient_accum_steps, epoch, global_step,
            local_rank, logger
        )
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch + 1} completed | Avg Loss: {epoch_loss:.4f}")
    
    # 保存最终模型
    if local_rank == 0:
        final_path = os.path.join(QATrainingConfig.output_dir, "final_model.pt")
        torch.save(model.module.state_dict(), final_path)
        logger.info(f"Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    main()