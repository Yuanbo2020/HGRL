import sys, os, argparse

# 这里的0是GPU id
import numpy as np

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *
from framework.utilities import cal_acc
from sklearn.metrics import r2_score
from sklearn import metrics


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    models = ['DNN', 'CNN', 'CNN_Transformer', 'PANN_fixed', 'PANN_fine_tuning', 'HGRL']
    model_index = models.index(model_type)
    using_models = [DNN, CNN, CNN_Transformer, PANN, PANN, HGRL]

    event_class = len(config.event_labels)

    if model_index==5:
        hidden_dim = 32
        out_dim = 64
        emb_dim = 64
        model = using_models[model_index](event_num=event_class, hidden_dim=hidden_dim,
                            out_dim=out_dim, in_dim=emb_dim, n_layers=3, emb_dim=emb_dim)
    else:
        model = using_models[model_index](event_class=event_class)
        # print(model)

    model_names = ['DNN.pth', 'CNN.pth', 'CNN_Transformer.pth', 'PANN_fixed.pth', 'PANN_fine_tuning.pth', 'HGRL.pth']
    result_dir = os.path.join(os.getcwd(), 'pretrained_models')
    model_path = os.path.join(result_dir, model_names[model_index])

    model_event = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_event['state_dict'])

    if config.cuda:
        model.cuda()

    dataset_path = os.path.join(os.getcwd(), 'Dataset')
    if model_index == 5:
        generator = DataGenerator_HGRL(dataset_path=dataset_path, normalization=True, emb_dim=emb_dim)
    else:
        generator = DataGenerator_DNN_CNN_CNN_Transformer(dataset_path=dataset_path, normalization=True)

    data_type = 'testing'
    generate_func = generator.generate_testing(data_type=data_type)

    # Forward
    if model_index == 5:
        dict = forward_HGRL(model=model, generate_func=generate_func, cuda=config.cuda)
    else:
        dict = forward_DNN_CNN_CNN_Transformer(model=model, generate_func=generate_func, cuda=config.cuda)

    # rate loss
    targets = dict['target']
    predictions = dict['output']
    mse_loss = metrics.mean_squared_error(targets, predictions)
    mae_loss = metrics.mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    print("ARP:\n\tmse_loss: {},  mae_loss: {}, r2: {}".format(mse_loss, mae_loss, r2))

    ###################################################################################################################
    # rate loss
    targets = dict['targets_event']
    predictions = dict['outputs_events']

    Acc = cal_acc(targets, predictions)
    print('AEC:\n\tAcc: ', Acc)



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















