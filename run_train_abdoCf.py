import os
from os.path import abspath, dirname
import yaml
import argparse
from datetime import datetime



import numpy as np
import matplotlib.pyplot as plt




def main():

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Config file')
    args = parser.parse_args()

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%H_%M_%S-%f_%b-%d-%Y")
    datestampStr = dateTimeObj.strftime("%m-%d-%Y")

    current_folder = abspath(dirname(__file__)) + "/"

    # Load config
    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh)

    seed = config['SEED']
    np.random.seed(seed)
    torch.manual_seed(seed)

    rules_list = config['DATA']['rules'].split(',') # list of rules in the dataset
    print(f'rules list: {rules_list}')
    col2rules = dict({i:rules_list[i] for i in range(len(rules_list))})
    rules2col_dict = dict({rule:i for i, rule in enumerate(rules_list)})

    train_data_folder = config['DATA']['train_data_folder'] # data folder
    folder_tag = train_data_folder.split('/')[-1] # tag to identify ranking matrix
    assert len(folder_tag) > 1

    folder_tag += '_seen-objs'

    num_epochs = 350

    # load ratings matrix, get matrix shape 
    ratings_matrix = np.load(config['MODEL']['train_ranking_matrix_file'])
    print('Ranking matrix, ', config['MODEL']['train_ranking_matrix_file'])
    num_pairs, num_schemas = ratings_matrix.shape

    # non negative indices (x, y)
    nonneg_indices_xy = np.nonzero(ratings_matrix >= 0)

    # flatten ratings matrix
    ratings_ravel = ratings_matrix.ravel() 
    nonneg_indices_ravel = np.nonzero(ratings_ravel >= 0)[0]

    assert all(np.array([x*num_schemas + y for x, y in zip(*nonneg_indices_xy)]) == nonneg_indices_ravel), \
        'issue with non negative indices'

    print('Total size of matrix {}, number of non-negative elements {}'\
            .format(num_pairs*num_schemas, len(nonneg_indices_ravel)))

    # numpy to tensor
    ratings_ravel = torch.tensor(ratings_ravel, 
                                    dtype=torch.float, 
                                    requires_grad=False)

    #hyperparameters
    lambda_reg = 1e-2
    learning_rate = 1e-2
    hidden_dimension = 3

    # report hyperparams
    print(f'Hidden dimension: {hidden_dimension}, Num pairs: {num_pairs}, Num schemas: {num_schemas}')

    model = OrganizeMyShelves(hidden_dimension, num_pairs, num_schemas, lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)    

    # Val dataset
    val_ratings_matrix = np.load(config['MODEL']['val_ranking_matrix_file'])
    # non negative indices (x, y)
    val_nonneg_indices_xy = np.nonzero(val_ratings_matrix >= 0)

    # flatten ratings matrix and convert to tensor
    val_ratings_ravel = val_ratings_matrix.ravel()
    val_ratings_ravel = torch.tensor(val_ratings_ravel, 
                                        dtype=torch.float, 
                                        requires_grad=False) 
    val_nonneg_indices_ravel = np.nonzero(val_ratings_ravel >= 0)[0]

    losses = []

    ep = 1
    while True:
        
        ratings_pred = model.forward()
        mse_loss, total_loss = \
            model.calculate_loss(ratings_pred, 
                                    ratings_ravel, 
                                    nonneg_indices_ravel,
                                    nonneg_indices_xy)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        save_mse_loss = mse_loss.detach().cpu().numpy()

        losses.append(save_mse_loss)

        if len(losses) >= 2:
            convergence_rate = (losses[-2] - save_mse_loss)/losses[-2]
        else:
            convergence_rate = 1

        print('Epoch: {}, Train loss: {}, Convergence: {}'.format(ep+1, save_mse_loss, convergence_rate))

        if convergence_rate < 1e-3:
            break

        with torch.no_grad():
            ratings_pred = model.forward()
            val_mse_loss, _ = model.calculate_loss(ratings_pred,
                                                    val_ratings_ravel,
                                                    val_nonneg_indices_ravel,
                                                    val_nonneg_indices_xy)

            print('Epoch: {}, Val loss: {}'.format(ep+1, val_mse_loss.cpu().numpy()))

        optimizer.zero_grad()
        ep += 1

    print(f'Initial train MSE loss: {losses[0]}')
    print(f'Final train MSE loss: {losses[-1]}')

    # save plot
    plt.plot(losses)
    plt.yticks(np.arange(0, 5, 0.2))
    plt.ylim(0, 5)
    plt.savefig(os.path.join(current_folder, f'train_loss_{folder_tag}.png'))

    # tensor to numpy 
    biases_obj_pair = model.biases_obj_pair.cpu().detach().numpy()
    biases_schema = model.biases_schema.cpu().detach().numpy()
    obj_preference_matrix = model.obj_preference_matrix.cpu().detach().numpy()
    schema_preference_matrix = model.schema_preference_matrix.cpu().detach().numpy()

    np.savez(os.path.join(current_folder, f'trained_matrix_cf_{folder_tag}_allrules_{timestampStr}.npz'),
                biases_obj_pair=biases_obj_pair,
                biases_schema=biases_schema,
                obj_preference_matrix=obj_preference_matrix,
                schema_preference_matrix=schema_preference_matrix
                )

if __name__ == '__main__':
    main()