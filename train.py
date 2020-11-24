import torch
import numpy as np
import os
import sys
import math
import logging
import json
from tqdm import tqdm, trange
from process_eicu import get_datasets
from utils import *
from graph_convolutional_transformer import GraphConvolutionalTransformer

from tensorboardX import SummaryWriter
import torchsummary as summary

    
def prediction_loop(args, model, dataloader, priors_datalaoder, description='Evaluating'):

    batch_size = dataloader.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()
    
    for data, priors_data in tqdm(zip(dataloader, priors_datalaoder), desc=description):
        data, priors_data = prepare_data(data, priors_data, args.device)
        with torch.no_grad():
            outputs = model(data, priors_data)
            loss = outputs[0].mean().item()
            logits = outputs[1]
        
        labels = data[args.label_key]
        
        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss]*batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
    
    if preds is not None:
        preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids)
    
    metrics['eval_loss'] = np.mean(eval_losses)
    
    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics['eval_{}'.format(key)] = metrics.pop(key)
    
    return metrics


def main():
    args = ArgParser().parse_args()
    set_seed(args.seed) 

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    
    logging.info("Arguments %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    tb_writer = SummaryWriter(log_dir=logging_dir)

    # Dataset handling
    datasets, prior_guides = get_datasets(args.data_dir, fold=args.fold)
    train_dataset, eval_dataset, test_dataset = datasets
    train_priors, eval_priors, test_priors = prior_guides
    train_priors_dataset = eICUDataset(train_priors)
    eval_priors_dataset = eICUDataset(eval_priors)
    test_priors_dataset = eICUDataset(test_priors)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    train_priors_dataloader = DataLoader(train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    eval_priors_dataloader = DataLoader(eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader = DataLoader(test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)

    if args.do_train:
        model = GraphConvolutionalTransformer(args)
        model = model.to(args.device)
        
        num_update_steps_per_epoch = len(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
        num_train_epochs = int(np.ceil(num_train_epochs))
        
        args.eval_steps = num_update_steps_per_epoch // 2

        #also try Adamax
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=args.eps)
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
        warmup_steps = max_steps // (1 / args.warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=max_steps)

        # if tb_writer:
        #     tb_writer.add_text('args', json.dumps(vars(args), indent=2, sort_keys=True))
        
        logger.info('***** Running Training *****')
        logger.info(' Num examples = {}'.format(len(train_dataloader.dataset)))
        logger.info(' Num epochs = {}'.format(num_train_epochs))
        logger.info(' Train batch size = {}'.format(args.batch_size))
        logger.info(' Total optimization steps = {}'.format(max_steps))

        epochs_trained = 0
        global_step = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        
        train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_pbar = tqdm(train_dataloader, desc='Iteration')
            for data, priors_data in zip(train_dataloader, train_priors_dataloader):
                model.train()
                data, priors_data = prepare_data(data, priors_data, args.device)

                # [loss, logits, all_hidden_states, all_attentions]
                outputs = model(data, priors_data)
                loss = outputs[0]
                
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                
                tr_loss += loss.detach()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                if (args.logging_steps > 0 and global_step % args.logging_steps==0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logging_loss_scalar = tr_loss_scalar
                    if tb_writer:
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(k, v, global_step)
                        tb_writer.flush()
                    output = {**logs, **{"step": global_step}}
                    print(output)
                
                if (args.eval_steps > 0 and global_step % args.eval_steps==0):
                    metrics = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
                    logger.info('**** Checkpoint Eval Results ****')
                    for key, value in metrics.items():
                        logger.info('{} = {}'.format(key, value))
                        tb_writer.add_scalar(key, value, global_step)
                    
                        
                epoch_pbar.update(1)
                if global_step >= max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if global_step >= max_steps:
                break
        
        train_pbar.close()
        if tb_writer:
            tb_writer.close()
            
        logging.info('\n\nTraining completed')


    eval_results = {}
    if args.do_eval:
        logger.info('*** Evaluate ***')
        logger.info(' Num examples = {}'.format(len(eval_dataloader.dataset)))
        eval_result = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('*** Eval Results ***')
            for key, value in eval_result.items():
                logger.info("{} = {}".format(key, value))
                writer.write('{} = {}'.format(key, value))
        eval_results.update(eval_result)

    if args.do_test:
        logging.info('*** Test ***')
        # predict
        test_result = prediction_loop(args, model, test_dataloader, test_priors_dataloader, description='Testing')
        output_test_file = os.path.join(args.output_dir, 'test_results.txt')
        with open(output_test_file, 'w') as writer:
            logger.info('**** Test results ****')
            for key, value in test_result.items():
                logger.info('{} = {}'.format(key, value))
                writer.write('{} = {}'.format(key, value))
        eval_results.update(test_result)


def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print(name, shape, param_size)
        total_params += param_size
    print(total_params)
        

if __name__ == "__main__":
    main()
    