import json
from pathlib import Path


class Config:
    def __init__(self, config_path=None):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config_name = config_path.split('/')[-1].split('.')[0]
        except:
            config_dict = {}

        # data
        self.dataset_name = config_dict.get('dataset_name', 'USPTO-50k')
        self.RawDataFile_path = config_dict.get('RawDataFile_path', './data/RawData')
        self.ProcessedDataFile_path = config_dict.get('ProcessedDataFile_path', './data/ProcessedData')
        # self.num_v_lgs = config_dict.get('num_v_lgs', 650)

        # RCI

        # neural network
        self.hidden_dimension = config_dict.get('hidden_dimension', 128)
        self.num_egat_layers = config_dict.get('num_egat_layers', 4)
        self.num_egat_heads = config_dict.get('num_egat_heads', 4)
        self.num_of_LayerHypergraph = config_dict.get('num_of_LayerHypergraph', 1)

        # ablation
        self.have_fp = config_dict.get('have_fp', True)
        self.residual = config_dict.get('residual', True)
        self.have_structure = config_dict.get('have_structure', True)

        # ppo
        self.gamma = config_dict.get('gamma', 0.99)
        self.eps_clip = config_dict.get('eps_clip', 0.2)
        self.value_coefficient = config_dict.get('value_coefficient', 0.5)
        self.entropy_coefficient = config_dict.get('entropy_coefficient', 0.01)
        self.num_imitation = config_dict.get('num_imitation', 8)

        # training
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.total_num_epoch = config_dict.get('total_num_epoch', 128)
        self.min_num_transitions = config_dict.get('min_num_transitions', 1024)
        self.num_epochs = config_dict.get('num_epochs', 4)
        self.batch_size = config_dict.get('batch_size', 512)
        self.save_at_num_update = config_dict.get('save_at_num_update', 100)

        # divice
        self.device = config_dict.get('device', 'cuda:0')

        # env
        self.trajectory_max_length = config_dict.get('trajectory_max_length', 4)

    def to_dict(self):
        return vars(self)

    def save(self, config_path):
        if Path(config_path).suffix == '':
            config_path += '.json'
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


