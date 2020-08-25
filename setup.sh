#setup the repository so the htp model can be trained or run

#first need to put together the preprocessed data for training (they are in pieces because of github's single file size limit)
cat ./shtp_data_preprocessed/shtp_data.tar.gz.parta* > shtp_data.tar.gz

#then extract the preprocessed data
tar -xvf shtp_data.tar.gz

#tell the user they can run the model file now
echo "Now you can run the command 'python3 ADI_model.py' to train the model"
