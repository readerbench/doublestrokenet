## DoubleStrokeNet: Bigram-Level Keystroke Authentication

The method analyzes the unique typing patterns of individuals to verify their identity while interacting with the keyboard, both virtual and hardware. Current deep learning approaches like TypeNet and TypeFormer focus on generating biometric signatures as embeddings for the entire typing sequence. The authentication process is defined using Euclidean distances between the new typing embedding and the saved biometric signatures. This paper proposes a new approach, called DoubleStrokeNet, to authenticate users through keystroke analysis using bigram embeddings. Instead of analyzing entire sequences, the model focuses on the temporal features of bigrams to learn user embeddings. The model employs a Transformer-based neural network to distinguish between bigrams and utilizes self-supervised learning to learn embeddings for bigrams and users.

`data_preproc` folder contains the code for preprocessing the data for the Aalto desktop and mobile datasets.

After preprocessing the data, the `main.py` script can be used to pretrain/finetune the model. The config file can be found in `./conf/config.yaml` where the data, model and training parameters can be set.
