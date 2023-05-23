import torch
import torch.nn.functional as F
import numpy as np

from framework.models_pytorch import move_data_to_gpu


def forward_DNN_CNN_CNN_Transformer(model, generate_func, cuda):
    outputs = []
    outputs_events = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_y_event ) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            linear_each_events, linear_rate = model(batch_x)

            batch_each_events = F.sigmoid(linear_each_events)

            outputs.append(linear_rate.data.cpu().numpy())

            outputs_events.append(batch_each_events.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    outputs_events = np.concatenate(outputs_events, axis=0)
    dict['outputs_events'] = outputs_events


    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def forward_HGRL(model, generate_func, cuda):
    outputs = []
    outputs_events = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_y_event, batch_graph, batch_y_semantic7) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            linear_each_events, linear_semantic7, linear_rate = model(batch_x, batch_graph)

            batch_each_events = F.sigmoid(linear_each_events)
            outputs_events.append(batch_each_events.data.cpu().numpy())

            outputs.append(linear_rate.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    outputs_events = np.concatenate(outputs_events, axis=0)
    dict['outputs_events'] = outputs_events

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event

    return dict








