# Reconciling-Robustness-and-Interpretability-in-Machine-Learning


### Installation in Ubuntu

        python3 -m venv venv
        source venv/bin/activate
        python -m pip install --upgrade pip
        python -m pip install wheel numpy
        sudo apt-get install libopenmpi-dev
        python -m pip install abcpy matplotlib

### To run Heston model samples

        python heston.py


### To run ABC Rejection for Heston model

        python heston_model.py
