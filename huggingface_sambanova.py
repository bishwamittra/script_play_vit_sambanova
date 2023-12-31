import torch
import random
import os
import numpy as np
from data import ChestXrayDataSet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from time import time
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score


# From sambaflow
import sambaflow.samba.utils as sn_utils
from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver



if(False): # compilation does not rely on this code

    path_image = "input/images/"
    train_df_path ="input/labels/train_list.txt"
    test_df_path = "input/labels/test_list.txt"
    val_df_path = "input/labels/val_list.txt"

num_epochs = 10
num_labels = 14
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 131
seed_everything(seed)

# The maximum image size in SambaNova: 384 x 384

if(False): # compilation does not rely on this code
    train_dataset = ChestXrayDataSet(data_dir=path_image,
                                        image_list_file=train_df_path,
                                        transform=transforms.Compose(
                                        [
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize
                                        ]
                                        ))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)

    val_dataset = ChestXrayDataSet(data_dir=path_image,
                                        image_list_file=val_df_path,
                                        transform=transforms.Compose(
                                        [
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)



    test_dataset = ChestXrayDataSet(data_dir=path_image,
                                        image_list_file=test_df_path,
                                        transform=transforms.Compose(
                                        [
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize
                                        ]
                                        ))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)


# Load ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
model.config.num_labels = num_labels  # Set the number of output classes
model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)




# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([num_labels]))
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# optimizer = samba.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

optimizer = samba.optim.AdamW(model.parameters(), lr=1e-4)
# quit()

def get_inputs():
    images_sn = samba.randn(batch_size, 3, 224, 224,
				name='image',
				batch_dim=0).bfloat16()
    labels_sn = samba.randint(num_labels, (batch_size, num_labels), # for multi-label classification
				name='label',
				batch_dim=0,
				dtype=torch.int16)


    return (images_sn, labels_sn)

samba.from_torch_model_(model.bfloat16())
inputs = get_inputs()

args = parse_app_args()
if args.command == 'run':
        raise NotImplementedError()
        train_loader, test_loader = prepare_dataloader(args)
        sn_utils.trace_graph(model, inputs, optimizer, pef=args.pef)
        train(args, model, train_loader)
        test(args, model, test_loader)
else:
    common_app_driver(args, model, inputs, optimizer, name='vit')


quit()

# Train the model and benchmark the wall clock time
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images).logits
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # for AUC-ROC

    with torch.no_grad():
        for batch in loader:
            # images, labels = batch['image'], batch['label']
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_auc = roc_auc_score(all_labels, all_probs, average='macro')
    val_sensitivity = recall_score(all_labels, all_preds, average='macro')
    val_precision = precision_score(all_labels, all_preds, average='macro')

    return val_loss, val_f1, val_auc, val_sensitivity, val_precision
    


# Start the timer
start_time = time()


for epoch in range(num_epochs):
    # training
    start_time_epoch = time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    end_time_epoch = time()
    # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Time taken: {end_time_epoch - start_time_epoch}")

    # validation performance
    start_time_evaluate_epoch = time()
    # val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
    val_loss, val_f1, val_auc, val_sensitivity, val_precision = evaluate(model, val_loader, criterion, device)
    end_time_evaluate_epoch = time()


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1 Score: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Val Sensitivity: {val_sensitivity:.4f}, Val Precision: {val_precision:.4f}, Training Time taken: {end_time_epoch - start_time_epoch}, Evaluation Time taken: {end_time_evaluate_epoch - start_time_evaluate_epoch}")


# Test performance
start_time_test = time()
test_loss, test_f1, test_auc, test_sensitivity, test_precision = evaluate(model, test_loader, criterion, device)
end_time_test = time()
print(f"\n\nFinal results: Test Loss: {test_loss:.4f}, test F1: {test_f1:.4f}, test AUC: {test_auc:.4f}, test Sensitivity: {test_sensitivity:.4f}, Test Precision: {test_precision:.4f}, Time taken: {end_time_test - start_time_test}")
# print(f"\n\nFinal results: Test Loss: {test_loss:.4f}, test F1: {test_f1:.4f}, Time taken: {end_time_test - start_time_test}")


# Stop the timer
end_time = time()

# Print the wall clock time
print(f"Training wall clock time: {end_time - start_time} seconds")

    
