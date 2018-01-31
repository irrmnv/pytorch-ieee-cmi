import os
import logging

import random
import numpy as np

import torch
import torch.nn.functional as F

checkpoint_dir = os.getcwd() + '/models'
cuda_is_available = torch.cuda.is_available()

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(torch.autograd.Variable(x.cuda(async=True), volatile=volatile))

def cuda(x):
    return x.cuda() if cuda_is_available else x

def train_and_validate(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    start_epoch,
    best_val_loss,
    experiment_name,
):
    if best_val_loss is None:
        best_val_loss = float('+inf')
    
    valid_losses = []
    lr_reset_epoch = start_epoch
    
    for epoch in range(start_epoch, epochs+1):
        train(
            train_data_loader,
            model,
            optimizer,
            loss_fn,
            epoch,
        )
        val_loss = validate(
            valid_data_loader,
            model,
            loss_fn,
        )
        valid_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss,
                },
                '{experiment_name}_{epoch}_{val_loss}.pth'.format(experiment_name=experiment_name, epoch=epoch, val_loss=val_loss),
                checkpoint_dir,
            )
            best_val_loss = val_loss
        scheduler.step(val_loss, epoch)
    return model

def train(train_loader, model, optimizer, criterion, epoch):
    losses = []
    
    model.train()
    
    logging.info('Epoch: {}'.format(epoch))
    for i, (inputs, O, targets) in enumerate(train_loader):
        inputs, O, targets = variable(inputs), variable(O), variable(targets)
        outputs = model(inputs, O)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        batch_size = inputs.size(0)
        (batch_size * loss).backward()
        optimizer.step()
        
        losses.append(loss.data[0])
        
        if i % 100 == 0:
            logging.info('Step: {}, train_loss: {}'.format(i, np.mean(losses[-100:])))
            
    train_loss = np.mean(losses)
    logging.info('train_loss: {}'.format(train_loss))

def validate(val_loader, model, criterion):
    accuracy_scores = []
    losses = []
    
    model.eval()
    
    for i, (inputs, O, targets) in enumerate(val_loader):
        inputs, O, targets = variable(inputs, volatile=True), variable(O), variable(targets)
        outputs = model(inputs, O)
        loss = criterion(outputs, targets)
        
        losses.append(loss.data[0])
        
        accuracy_scores += list(targets.data.cpu().numpy() == np.argmax(outputs.data.cpu().numpy(), axis=1))
        
    valid_loss, valid_accuracy = np.mean(losses), np.mean(accuracy_scores)
    logging.info('valid_loss: {}, valid_acc: {}'.format(valid_loss, valid_accuracy))
    return valid_loss

def save_checkpoint(state, filename, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)