import os
import torch
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def train_step(args, plm, pvm, encoder, loss_function, optimizer, dataloader):
    plm.train()
    pvm.train()
    encoder.train()
    loss_total = 0
    for batch_index, data in enumerate(dataloader):
        input_ids = data['input_ids'].to(args.DEVICE)
        attention_mask = data['attention_mask'].to(args.DEVICE)
        pixel_values = data['pixel_values'].to(args.DEVICE)
        labels = data['labels'].to(args.DEVICE)
        plm_logit = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        pvm_logit = pvm(pixel_values)
        logit = encoder(plm_logit, pvm_logit)
        loss = loss_function(logit, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss
        if (batch_index+1)%args.PRINT_BATCH == 0:
            print(f'Loss after {batch_index+1} batches: {loss:.5f}')
    loss_average = loss_total / len(dataloader)
    return loss_total, loss_average

def test_step(args, plm, pvm, encoder, dataloader):
    plm.eval()
    pvm.eval()
    encoder.eval()
    labels_true, labels_pred = [], []
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(args.DEVICE)
            attention_mask = data['attention_mask'].to(args.DEVICE)
            pixel_values = data['pixel_values'].to(args.DEVICE)
            labels = data['labels'].to(args.DEVICE)
            plm_logit = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            pvm_logit = pvm(pixel_values)
            logit = encoder(plm_logit, pvm_logit)
            labels_true.extend(labels.tolist())
            labels_pred.extend(logit.argmax(dim=-1).tolist())
    return labels_true, labels_pred

def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                               labels=tuple(labels_to_ids.values()), zero_division=0.0, digits=5)
    cls_report_dict = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                                    labels=tuple(labels_to_ids.values()), zero_division=0.0, output_dict=True)
    accuracy, macro_f1 = cls_report_dict['accuracy'], cls_report_dict['macro avg']['f1-score']
    return cls_report, accuracy, macro_f1

def plot_confmat(args, labels_true, labels_pred, ids_to_labels, labels_to_ids):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred,
                                                           labels=list(ids_to_labels.keys()),
                                                           display_labels=labels_to_ids,
                                                           xticks_rotation='vertical')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(args.FIG_SIZE)
    fig.set_figheight(args.FIG_SIZE)
    plt.show()

def save_model(pvm, plm, encoder, optimizer, path, model_name):
    save_path = path / model_name
    print(f'** Saving model to: {save_path} **')
    state = {"pvm": pvm.state_dict(),
             "plm": plm.state_dict(),
             "encoder": encoder.state_dict(),
             "optimizer": optimizer.state_dict()}
    torch.save(state, save_path)

# function to record all the user's input arguments (hyperparams in some cases) in a .txt file
def log_arguments(args, path):
    with open(path / args.INFO_FILE, "w") as f:
        f.write(f'Arguments:\n')
        f.write('------------------------\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        f.write('------------------------\n')

# function to record the progress (sklearn classification report) 
# after each epoch in a .txt file (should be in the same file as log_hyperparams)
def log_progress(args, epoch, loss_total, loss_average, path, cls_report=None):
    with open(path / args.INFO_FILE, "a") as f:
        f.write(f'\nepoch {epoch}:\n')
        f.write(f'Total loss: {loss_total:.5f} | Average loss: {loss_average:.5f}\n')
        if cls_report:
            f.write(f'Classification report:\n{cls_report}')

def export_prediction(labels_pred, ids_to_labels, path, name='prediction.csv'):
    pd.DataFrame({'prediction': map(ids_to_labels.get, labels_pred)}).to_csv(path / name, index=False)

def test_best_model(args, labels_to_ids, dataloader, model_path, plm, pvm, encoder):
    saved_models = sorted(float(model[:-3]) for model in os.listdir(model_path) if model.split('.')[-1] == 'pt')
    state = torch.load(model_path / f'{saved_models[-1]}.pt', weights_only=False)
    plm.load_state_dict(state['plm'])
    pvm.load_state_dict(state['pvm'])
    encoder.load_state_dict(state['encoder'])
    labels_true, labels_pred = test_step(args, plm, pvm, encoder, dataloader)
    return labels_true, labels_pred