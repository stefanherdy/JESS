# JESS
Joint Energy-Based Semantic Segmentation

![segmantation image](https://github.com/stefanherdy/JESS/blob/main/img/seg.png?raw=true)


Installation

$ git clone https://github.com/stefanherdy/JESS.git

Usage

    - Run train_jess.py with "python train_jess.py".
        You can specify the following parameters:
        --batch_size, type=int, default=8, help="Batch Size"
        --learnrate, type=int, default=0.0001, help='learn rate of optimizer'
        --p_x_weight, type=int, default=0.01, help='weight of energy based optimization'
        --optimizer, choices=['sgd', 'adam'], default='adam'
        --eval_every, type=int, default=1, help="Epochs between evaluation"
        --print_every, type=int, default=1, help="Epochs between print"
        --ckpt_every, type=int, default=20, help="Epochs between checkpoint save"
        --energy, type=bool, default=True, help="Set p(x) optimization on(True)/off(False)"
        --num_classes, type=int, default=8, help="Number of classes"
        --num_tests, type=int, default=10, help="Number of tests"
        --test, choices=['norm', 'jess'], default='norm', help="Normal test or Joint Energy-Based Sematic Segmentation"
        --set, choices=['usa', 'john_handy', 'john_cam'], default='norm', help="Dataset"

        Example usage:
        "python train_jess.py --test norm --set usa --learnrate 0.00001 --batch_size 16

        
    - To evaluate the model run evaluate_model.py with "python evaluate_model.py".
        You can specify the following parameters:
        --test, choices=['norm', 'jess'], default='norm', help="Normal test or Joint Energy-Based Sematic Segmentation"
        --set, choices=['usa', 'john_handy', 'john_cam'], default='norm', help="Dataset"
        --num_classes, type=int, default=8, help="Number of classes"
        --batch_size, type=int, default=8, help="Batch Size"

        Example usage:
        "python evaluate_model.py --test norm --set usa 

        Make sure you performed the training before, so that the models can be loaded for evaluation.

        
    - You can run neighbors.py to run the neighbor analysis with "python neighbors.py".
        You can specify the following parameters:
        --test, choices=['norm', 'jess'], default='norm', help="Normal test or Joint Energy-Based Sematic Segmentation"
        --set, choices=['usa', 'john_handy', 'john_cam'], default='norm', help="Dataset"

        Example usage:
        "python neighbors.py --test norm --set usa 

License

This project is licensed under the MIT License. ©️ 2023 Stefan Herdy
