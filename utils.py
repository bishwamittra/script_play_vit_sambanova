import logging
import os
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score


def get_logger(logger_path):
    logging.basicConfig(
        filename=logger_path,
        # filename='/home/qinbin/test.log',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger 

def set_path(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # prepare the save path
    save_tag = f'vit-chestxray14-seed{args.seed}-ep{args.epoch}-bs{args.batch_size}-lr{args.lr}' 

    # if args.save_results or args.save_curves:
    exp_seq_path = os.path.join(args.save_root, 'exp_seq.txt')
    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq=0
        exp_seq = str(exp_seq)
        file.write(exp_seq)
        file.close
        save_tag = 'exp_' + exp_seq + '_' + save_tag
    else:
        file = open(exp_seq_path, 'r')
        exp_seq = int(file.read())
        exp_seq += 1
        exp_seq = str(exp_seq)
        save_tag = 'exp_' + exp_seq + '_' + save_tag
        file = open(exp_seq_path, 'w')
        file.write(exp_seq)
        file.close()
    args.exp_seq = exp_seq
    args.save_path = os.path.join(args.save_root, save_tag)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.config_path = os.path.join(args.save_path, 'config.json')
    args.logger_path = os.path.join(args.save_path, 'exp_log.log')   
   
    return args

# Train the model and benchmark the wall clock time
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, eval_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # for AUC-ROC

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(eval_loader)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_auc = roc_auc_score(all_labels, all_probs, average='weighted')
    val_sensitivity = recall_score(all_labels, all_preds, average='weighted')
    val_precision = precision_score(all_labels, all_preds, average='weighted')

    return val_loss, val_f1, val_auc, val_sensitivity, val_precision