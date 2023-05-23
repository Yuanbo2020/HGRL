import torch

####################################################################################################

cuda = 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64
batch_size = 64
epoch = 100
lr_init = 1e-3



event_labels = ['Aircraft', 'Bells', 'Bird tweet', 'Bus', 'Car', 'Children', 'Construction',
                'Dog bark', 'Footsteps', 'General traffic', 'Horn', 'Laughter', 'Motorcycle', 'Music',
                'Non-identifiable', 'Other', 'Rail', 'Rustling leaves', 'Screeching brakes', 'Shouting',
                'Siren', 'Speech', 'Ventilation', 'Water']

sort_object_event_labels = ['Aircraft', 'Bus', 'Car', 'General traffic', 'Motorcycle', 'Rail', 'Screeching brakes',
                'Bells', 'Music',
                'Bird tweet', 'Dog bark',
                'Children', 'Laughter',  'Speech', 'Shouting', 'Footsteps',
                'Siren', 'Horn',
                'Rustling leaves', 'Water',
                'Construction', 'Non-identifiable', 'Other', 'Ventilation', ]

source_to_sort_indices = [sort_object_event_labels.index(each) for each in event_labels]
# print(source_to_sort_indices)

subject_labels = ['Vehicle',
                  'Music',
                  'Animals',
                  'Human sounds',
                  'Alarm',
                  'Natural sounds',
                  'Other'
                  ]


endswith = '.pth'

cuda_seed = None
