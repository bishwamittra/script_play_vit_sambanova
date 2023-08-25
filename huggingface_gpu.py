import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from time import time, strftime
import json
# from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

from data import ChestXrayDataSet
from utils import *

def argparser():
    parser = argparse.ArgumentParser(description='centralized sgd baseline')

    # default args - data set and model
    parser.add_argument('--device', type=str, default='cuda', help='choose from cuda, cpu, mps')
    parser.add_argument('--seed', type=int, default=42, help='set a seed for reproducability')
    parser.add_argument('--lr', type=float, default=0.01, help='server learning rate for updating global model by the server')
    parser.add_argument('--batch_size', type=int, default=64, help='server batch size for training global model')
    parser.add_argument('--epoch', type=int, default=50, help='server epochs to train global model with synthetic data')
    parser.add_argument('--save_root', type=str, default='./results/', help='path to save results')
    parser.add_argument('--one_gpu', action='store_true', default=False, help='use only one gpu')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to fine-tune pretrained model weights')

    args = parser.parse_args()

    return args

    
def main(args, logger):

    device = args.device
    path_image = args.path_image
    train_df_path = args.train_df_path
    test_df_path = args.test_df_path
    val_df_path = args.val_df_path

    num_epochs = args.epoch
    num_labels = 14
    batch_size = args.batch_size
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    # max 384 x 384

    train_dataset = ChestXrayDataSet(
        data_dir=path_image,
        image_list_file=train_df_path,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
        )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    val_dataset = ChestXrayDataSet(
        data_dir=path_image,
        image_list_file=val_df_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
        )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    test_dataset = ChestXrayDataSet(
        data_dir=path_image,
        image_list_file=test_df_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
        )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
        )


    # Load ViT model and feature extractor
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    if args.pretrain:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        model.config.num_labels = num_labels  # Set the number of output classes
        model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)
    else:
        model = ViTForImageClassification(config=ViTConfig(num_labels=14)) # WY: this should work

    # set multi-gpu parallel
    if not args.one_gpu:
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    model.to(device)
    logger.info('>> Model initialized')

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([num_labels])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_time_all_epochs, total_time_train = 0, 0
    best_loss = 1e10 
    logger.info(f">> Starting experiment {args.exp_seq}")
    training_time_start = time()
    for epoch in range(num_epochs):
        # training
        start_time_epoch_train = time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        end_time_epoch_train = time()
        epoch_time_train = end_time_epoch_train - start_time_epoch_train
        total_time_train += epoch_time_train

        # validation performance
        start_time_epoch_eval = time()
        val_loss, val_f1, val_auc, val_sensitivity, val_precision = evaluate(model, val_loader, criterion, device)
        end_time_epoch_eval = time()
        epoch_time_eval = end_time_epoch_eval - start_time_epoch_eval
        epoch_time_all = epoch_time_train + epoch_time_eval


        # save checkpoint
        if val_loss<best_loss:
            best_loss=val_loss
            best_mdl=model.state_dict() 
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_mdl,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(args.save_path, 'checkpoint.pt'))
            logger.info(f'Epoch {epoch:3d}: >> Checkpoint saved')

        
        # print the epoch information
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Completed in {epoch_time_all/60:.2f} mins")
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Training time: {epoch_time_train/60:.2f} mins")
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Val Loss: {val_loss:.4f}, Checkpoint val loss: {best_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Val F1 Score: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Val Sensitivity: {val_sensitivity:.4f}, Val Precision: {val_precision:.4f}")
    
    training_time_end=time()
    total_time_all_epochs = training_time_end - training_time_start
    logger.info(f">> Training completed.")
    # logger.info(f">> Training completed, time elapsed: {total_time_all_epochs/60:.2f} mins")
   
    # Test performance
    logger.info(">> Testing in progress...")
    model.load_state_dict(best_mdl) # load the best model states from checkpoint
    start_time_test = time()
    test_loss, test_f1, test_auc, test_sensitivity, test_precision = evaluate(model, test_loader, criterion, device)
    end_time_test = time()
    total_time_eval = end_time_test - start_time_test
    logger.info(f">> Final results: Test Loss: {test_loss:.4f}, test F1: {test_f1:.4f}, test AUC: {test_auc:.4f}, test Sensitivity: {test_sensitivity:.4f}, Test Precision: {test_precision:.4f}")
    logger.info(f">> Testing time elapsed: {total_time_eval/60:.2f} mins")

    # summary
    end_time_stamp = strftime('%Y-%m-%d %H:%M:%S')
    logger.info("**************************************************************")
    logger.info(f">> Experiment {args.exp_seq} completes at {end_time_stamp}")
    logger.info(f">> Total time used for training: {total_time_train/60:.2f} mins")
    logger.info(f">> Total time used over {num_epochs} epochs: {total_time_all_epochs/60:.2f} mins")



if __name__ == '__main__':

    # read arguments
    args = argparser()
    set_path(args)

    # save configuration json file
    with open(args.config_path, 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        f.close()

    # set the logger
    logger = get_logger(args.logger_path)

    # set seed for this experiment trial
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)     
    
    # specify the data path
    args.path_image = "input/images/"
    args.train_df_path ="input/labels/train_list.txt"
    args.test_df_path = "input/labels/test_list.txt"
    args.val_df_path = "input/labels/val_list.txt"

    # set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # run the main function to train model
    main(args, logger)

