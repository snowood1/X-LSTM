----------------------------------------------------------------------------------------
Thank you for taking the time to review our code and datasets. This readme.txt describes how to run our proposed method. 
Before the demonstration, we first take a look at the structure of our code and datasets. The folder structure as following:

	+X-LSTM						# A Cross-lingual BiLSTM model for both pentacode classfication and source-target labelling (NER).
         +datasets				# training/testing datasets for classification task and NER task.
         +embedding				# multingual word embeddings
         +fasttext_multingual	# This contains the tool to align multingual embeddings into a single vector space
          			
	     models.py				# The implementation of model 1 (BiLSTM-FC for classification) and model 2 (BilSTM-CRF for labelling).
	     inference.py			# Labelling inference algorithm for model 2
	     utils.py  				# utils functions.
	     trainer.py 			# The implementation of the run() function.
	     main.py				# The main api interface to use run() from trainer.py.
	     ...				
		+readme.txt

Our code is written by Python3.6 and pytorch in GPU version. We assume your Operating System is GNU/Linux-based.
However, if you have MacOS or MacBook, it will be okay. If you use Windows, we recommend
that you use PyCharm to run our code. The dependencies of our programs is Python3.6 in miniconda.

Notice: Since our code depends on Python3.6, for people who are not familiar with
GNU/Linux environment, it may be difficult to run our code. However, we will provide
our pip command if the paper got accepted.

----------------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. It has two steps:
    1.  activate a new environment with Python3.6
    	conda create --name py36 python=3.6
    	source activate py36
    
    2. install torch and other packages
    	pip3 install torch torchvision torchaudio
    	pip3 install scikit-learn pathlib pandas

After set up above 2 steps, you are ready to run our code. 
----------------------------------------------------------------------------------------
Please run main.py to run a simple demo.

----------------------------------------------------------------------------------------
Notice: Since our programs are written in GPU version. Some programs above are time-cost or might occurs some unknown bugs if you only use one cpu. 
A better way is to test them in GPU. After above steps, you should be able to reproduce our results
reported  in our paper. If you cannot reproduce, please email: --@--.
----------------------------------------------------------------------------------------