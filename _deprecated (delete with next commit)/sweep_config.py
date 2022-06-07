pretrain_sweep_config = {
    'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
    'metric': {  # This is the metric we are interested in maximizing
      'name': 'MRR',
      'goal': 'maximize'   
    },
    # Paramters and parameter values we are sweeping across
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'epochs':{
            'values': [2, 3, 4]
        },
        'pretraining_contrastive_loss_fn': {
            'values': ["TripletMarginLoss", "InfoNceLoss"]
        },
        'speech_output_pooling_strategy': {
            'values': ["mean", "pooling_layer"]
        },
        'train_last_n_layers': {
            'vlues': [1, 2]
        }
    }
}