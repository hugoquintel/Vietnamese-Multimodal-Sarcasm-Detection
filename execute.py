# external modules
import os
import sys
import tqdm
import timm
import torch
import random
import pathlib
import matplotlib
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForTokenClassification

# created modules
from args import get_arguments
from model import MultimodalEncoder
from data_setup import make_dummy_data, preprocess_data, get_labels, SarcasmData
from utils import train_step, test_step, get_metrics, plot_confmat, export_prediction, \
                  save_model, log_arguments, log_progress, test_best_model

def run():
    # get user's input arguments
    args = get_arguments()

    font = {'size': args.FONT_SIZE}
    matplotlib.rc('font', **font)

    # set random seed (based on pytorch lightning set everything 
    # https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything)
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.cuda.manual_seed_all(args.RANDOM_SEED)

    # create some utility folders
    if args.SAVE_MODEL:
        model_path = pathlib.Path(args.SAVE_PATH)
        model_path.mkdir(parents=True, exist_ok=True)
    if args.EXPORT_PREDICTION:
        pred_path = pathlib.Path(args.PREDICTION_PATH)
        pred_path.mkdir(parents=True, exist_ok=True)
    info_path = pathlib.Path(args.INFO_PATH)
    info_path.mkdir(parents=True, exist_ok=True)
    
    # Handling data (import -> preprocess -> dataloader)
    data_path = pathlib.Path(args.DATA_PATH)
    if args.USE_DUMMY:
        dummy_path = pathlib.Path(args.DUMMY_PATH)
        make_dummy_data(data_path, args.DUMMY_DATASET, dummy_path, args.DUMMY_SAMPLES, dev_size=args.DUMMY_DEV_SIZE, shuffle=True)
        data_path = pathlib.Path(args.DUMMY_PATH)

    segmenter_tokenizer = AutoTokenizer.from_pretrained(args.SEG_PLM)
    segmenter_model = AutoModelForTokenClassification.from_pretrained(args.SEG_PLM).to(args.DEVICE)
    segmenter_tokenizer.model_max_length = args.PLM_MAX_TOKEN
    segmenter = pipeline("token-classification", model=segmenter_model, tokenizer=segmenter_tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    print('Handling train dataset...')
    train_df = preprocess_data(args, data_path, 'train', tokenizer, segmenter)
    print('Finshed train dataset')
    print('\nHandling dev dataset...')
    dev_df = preprocess_data(args, data_path, 'dev', tokenizer, segmenter)
    print('Finshed dev dataset')

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(train_df['labels']).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(dev_df['labels']).astype(int)

    pvm = timm.create_model(args.PVM, pretrained=True, num_classes=0).to(args.DEVICE)
    plm = AutoModel.from_pretrained(args.PLM, return_dict=True).to(args.DEVICE)
    encoder = MultimodalEncoder(plm.config, args.PLM_OUTPUT_SIZE, args.PVM_OUTPUT_SIZE, labels_to_ids).to(args.DEVICE)

    train_data = SarcasmData(pvm.default_cfg, train_df, data_path, 'train')
    dev_data = SarcasmData(pvm.default_cfg, dev_df, data_path, 'dev')

    no_workers = os.cpu_count()
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH, shuffle=True,
                                  pin_memory=True, num_workers=no_workers)
    dev_dataloader = DataLoader(dev_data, batch_size=args.DEV_BATCH, shuffle=False,
                                pin_memory=True, num_workers=no_workers)
    
    optimizer_map = {'ASGD': optim.ASGD, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad, 'Adam': optim.Adam,
                     'AdamW': optim.AdamW, 'Adamax': optim.Adamax, 'LBFGS': optim.LBFGS, 'NAdam': optim.NAdam, 'RAdam': optim.RAdam,
                     'RMSprop': optim.RMSprop, 'Rprop': optim.Rprop,'SGD': optim.SGD, 'SparseAdam': optim.SparseAdam}
    
    train_params = ({'params': pvm.parameters(), 'lr': args.PVM_LR},
                    {'params': plm.parameters(), 'lr': args.PLM_LR},
                    {'params': encoder.parameters(), 'lr': args.ENCODER_LR})

    optimizer = optimizer_map[args.OPTIMIZER](train_params)
    loss_function = nn.CrossEntropyLoss()

    print(f'Number of samples in train set: {len(train_data)}')
    print(f'Number of samples in dev set: {len(dev_data)}')
    print(f'Number of train batches: {len(train_dataloader)}')
    print(f'Number of dev batches: {len(dev_dataloader)}')
    print(f'Labels to ids: {labels_to_ids}')
    print(f'Ids to labels: {ids_to_labels}\n')

    if args.DEVICE == 'cuda':
        if torch.cuda.device_count() > 1:
            print('--Using multiple GPUs to train--\n')
            pvm = torch.nn.DataParallel(pvm)
            plm = torch.nn.DataParallel(plm)
            encoder = torch.nn.DataParallel(encoder)
        else: print('--Using single GPU to train--\n')
    else: print('--No GPU detected, using CPU to train--\n')

    print(f'Arguments:')
    print('------------------------')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('------------------------')
    log_arguments(args, info_path)

    best_accuracy, best_macro_f1, best_epoch = 0, 0, 0
    for epoch in tqdm.trange(args.EPOCHS, file=sys.stdout):
        print(f'\n\nEpoch {epoch}:')
        # train
        print('-----------')
        loss_total, loss_average = train_step(args, plm, pvm, encoder, loss_function, optimizer, train_dataloader)
        print(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}')
        print('-----------')
        # test
        labels_dev_true, labels_dev_pred = test_step(args, plm, pvm, encoder, dev_dataloader)
        if args.GET_METRICS:
            cls_report, accuracy, macro_f1 = get_metrics(labels_dev_true, labels_dev_pred, labels_to_ids)
            print('[+] METRICS:')
            print(f'Classification report:\n{cls_report}')
            log_progress(args, epoch, loss_total, loss_average, info_path, cls_report)
            if args.PLOT_CONFMAT:
                plot_confmat(args, labels_dev_true, labels_dev_pred, ids_to_labels, labels_to_ids)
            if args.SAVE_MODEL:
                if macro_f1 > best_macro_f1:
                    best_accuracy, best_macro_f1, best_epoch = accuracy, macro_f1, epoch
                    save_model(pvm, plm, encoder, optimizer, model_path, f"{round(best_macro_f1, 4)}.pt")
                    saved_models = sorted(float(model[:-3]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
                    if len(saved_models) > args.MODELS_LIMIT:
                        os.remove(model_path / f'{saved_models[0]}.pt')
        else:
            log_progress(args, epoch, loss_total, loss_average, info_path)
            if args.SAVE_MODEL:
                save_model(pvm, plm, encoder, optimizer, model_path, f'epoch_{epoch}.pt')
                saved_models = sorted(int(model[:-3].split('_')[1]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
                if len(saved_models) > args.MODELS_LIMIT:
                    os.remove(model_path / f'epoch_{saved_models[0]}.pt')
        
        if args.EXPORT_PREDICTION:
            if args.PREDICTION_PER_EPOCH:
                export_prediction(labels_dev_pred, ids_to_labels, pred_path, f'epoch_{epoch}.csv')
            else:
                export_prediction(labels_dev_pred, ids_to_labels, pred_path)

    if args.TEST_BEST_MODEL and args.GET_METRICS and args.SAVE_MODEL:
        print('\n\n\n******TESTING THE BEST MODEL******')
        labels_dev_true, labels_dev_pred = test_best_model(args, labels_to_ids, dev_dataloader, model_path, plm, pvm, encoder)
        cls_report, _, _ = get_metrics(labels_dev_true, labels_dev_pred, labels_to_ids)
        print('[+] METRICS:')
        print(f'Classification report:\n{cls_report}')
        if args.PLOT_CONFMAT:
            plot_confmat(args, labels_dev_true, labels_dev_pred, ids_to_labels, labels_to_ids)
        if args.EXPORT_PREDICTION:
            export_prediction(labels_dev_pred, ids_to_labels, pred_path, 'best_prediction.csv')
        print('**************FINISH**************')