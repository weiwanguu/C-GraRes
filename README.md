# Cross-Domain Tactile Object Recognition

This repository contains the published code for raw tactile cross-domain object recognition across 25 object classes. This public release corresponds to the experimental setup adopted in the accompanying manuscript.

## Repository Structure

```text
underwater-tactile-object-recognition/
|- main-R3-MS+SE-raw.py
|- main-CNN-raw-light.py
|- main-LSTM-raw-light.py
|- main-MLP-raw-light.py
|- raw_sequence_utils.py
|- requirements.txt
`- README.md
```



## Environment

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

Run one of the following scripts from the repository root:

```bash
python main-R3-MS+SE-raw.py
python main-CNN-raw-light.py
python main-LSTM-raw-light.py
python main-MLP-raw-light.py
```

All released scripts are configured for 30 training epochs.

## Notes

- Input tensor format for the models is `Batch x Channel x Length`.
- Training/validation/test splitting follows a stratified 6:2:2 ratio at the sample level before window generation.
- The loss-curve plotting scale is unified across models for direct visual comparison.
