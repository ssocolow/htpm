# htpm
This is a project to develop a human trajectory prediction model/method.  Inspired by https://github.com/xuehaouwa/SS-LSTM
(shtp means standard human trajectory prediction)  

We would like to thank the [creators of the ETH dataset](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/pellegrini09iccv.pdf) and the [creators of the UCY dataset](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2007.01089.x).

## Quick Start
Here is a google collab file to explore: 
https://colab.research.google.com/drive/1HQgejgDW-bQ4lh8iklCOmB7Q8fzEEOVw?usp=sharing

Or you can run it locally by running these commands:  
  
Download the files  
`git clone https://github.com/ssocolow/htpm.git`  
  
Go into the directory  
`cd htpm`  
  
Make the setup script executable (you may not have to do this step, do if the next step returns a permissions error)  
`chmod +x setup.sh`  
  
Run the setup script  
`./setup.sh`  
  
Train the model  
`python3 ADI_model.py`  

## Background
These papers lead up to this work
https://docs.google.com/document/d/1kWWwZU0jlQkvf-aBQ3mmEMIbwEpNAbTCfJpj0H_DWvU/edit?usp=sharing  

"SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction" by Hao Xue, Du Q. Huynh, and Mark Reynolds: https://www.researchgate.net/profile/Du_Huynh/publication/2269555_Self-Calibrating_a_Stereo_Head_An_Error_Analysis_in_the_Neighbourhood_of_Degenerate_Configurations/links/5c03ccb0a6fdcc1b8d502965/Self-Calibrating-a-Stereo-Head-An-Error-Analysis-in-the-Neighbourhood-of-Degenerate-Configurations.pdf

## Disclaimer
The processed data may be wrong
