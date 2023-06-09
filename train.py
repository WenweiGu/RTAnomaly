import torch
from CVAE import loss_function
import torch.optim as optim
from model import RTAnomaly
from dataloader import load_dataset, get_dataloaders, get_data_dim
from data_preprocess import normalize, generate_windows, minmax_score
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from evaluate import get_anomaly_score
import numpy as np
from evaluate import compute_prediction, compute_binary_metrics

# ['machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5', 'machine-1-6', 'machine-1-7',
#                'machine-1-8']

# ['machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5', 'machine-2-6', 'machine-2-7',
#                'machine-2-8', 'machine-2-9']

# ['machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7',
#                'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11']

# ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9']
# ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-11', 'D-12', 'D-13']
# ['E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13']
# ['F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-3', 'P-4', 'P-7']
# ['C-1', 'C-2', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8']
# ['M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15']
# ['S-2', 'T-4', 'T-5', 'T-8', 'T-9', 'T-12', 'T-13']

params = {
    'data_root': "./datasets/ASD",
    'train_postfix': "train.pkl",
    'test_postfix': "test.pkl",
    'test_label_postfix': "test_label.pkl",
    'dim': 19,
    'entity': ['omi-1', 'omi-2', 'omi-3', 'omi-4', 'omi-5', 'omi-6', 'omi-7', 'omi-8', 'omi-9', 'omi-10', 'omi-11',
               'omi-12'],
    'valid_ratio': 0,
    'normalize': "minmax",
    'window_size': 20,
    'stride': 5,
    'batch_size': 32,
    'num_workers': 0,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'gnn_dim': 128,
    'pooling_ratio': 0.5,
    'threshold': 0.5,
    'dropout': 0.5,
    'filters': [256, 256, 256],
    'kernels': [8, 5, 3],
    'dilation': [1, 2, 4],
    'layers': [50, 10],
    'gru_dim': 128,
    'epoch': 50,
    'lr': 1e-4,
    'wd': 1e-3,
    'recon_filter': 5,
    'hidden_size': 100,
    'latent_size': 10,
    'cof': 0.5
}


for item in params['entity']:
    logging.info("Fitting dataset: {}".format(item))
    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entity"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
    )

    data_dict = normalize(data_dict, method=params["normalize"])

    # sliding windows
    window_dict = generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"]
    )

    train_windows = window_dict[item]['train_windows']
    test_windows = window_dict[item]['test_windows']
    test_labels = window_dict[item]['test_label'][:, -1].reshape(-1, 1)

    train_loader, _, test_loader = get_dataloaders(
        train_windows,
        test_windows,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]
    )

    encoder = RTAnomaly(
        ndim=params['dim'],
        len_window=params['window_size'],
        gnn_dim=params['gnn_dim'],
        pooling_ratio=params['pooling_ratio'],
        threshold=params['threshold'],
        dropout=params['dropout'],
        filters=params['filters'],
        kernels=params['kernels'],
        dilation=params['dilation'],
        layers=params['layers'],
        gru_dim=params['gru_dim'],
        device=params['device'],
        recon_filter=params['recon_filter'],
        hidden_size=params['hidden_size'],
        latent_size=params['latent_size']
    )

    encoder.to(params['device'])

    optimizer = optim.Adam(encoder.parameters(),
                           lr=params['lr'], weight_decay=params['wd'])

    train = True
    test = True

    if train:
        encoder.train()
        for epoch in range(params['epoch']):
            loss = 0
            for i, x in enumerate(tqdm(train_loader)):
                if x.shape[0] == 1:
                    continue

                x = x.to(params['device'])  # 先放GPU上
                x = x.permute(0, 2, 1)
                train_label = torch.zeros((x.shape[0], 1)).to(params['device'])

                optimizer.zero_grad()
                x_recon, recon_embed, embed, mu, log_var = encoder(x, train_label)

                # loss 部分可以加入别的部分, 有一定作用
                loss_train = loss_function(x, x_recon, recon_embed, embed, mu, log_var, cof=params['cof'])
                loss += loss_train

                loss_train.backward()
                optimizer.step()

            loss /= train_loader.__len__()
            print(f'Training loss for epoch {epoch} is: {float(loss)}')

            torch.save(encoder.state_dict(), './save/checkpoint_' + item + '.pth')

    if test:
        logging.info("Finish dataset: {}".format(item))
        encoder.load_state_dict(torch.load('./save/checkpoint_' + item + '.pth'))
        encoder.eval()

        score = minmax_score(get_anomaly_score(test_loader, encoder, params['device'], 1))
        np.save(f'./score/score_{item}.npy', score)

        plt.plot(test_labels)
        plt.plot(score)
        plt.savefig(f'./save/checkpoint_{item}.jpg', dpi=600)
        plt.close()

    eval_score = np.load(f'./score/score_{item}.npy')
    test_labels = test_labels.flatten()
    pred, pred_adjust, _ = compute_prediction(eval_score, test_labels).values()

    print(f'Results for {item}:' + str(compute_binary_metrics(pred_adjust, test_labels)))
