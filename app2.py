from flask import Flask, render_template, jsonify, request
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import scipy.signal
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import mlab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.io
import logging
import scipy.signal 
app = Flask(__name__,template_folder='templates')
def generate_base64_image(plt):
    output = io.BytesIO()
    plt.savefig(output, format='png')
    plt.close()
    output.seek(0)
    return base64.b64encode(output.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    global m
    if m:
        # Access specific data from 'm'
        some_data = m['your_key']
        return jsonify({'data': some_data})
    else:
        return jsonify({'error': 'No data loaded yet.'}), 404










# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the global variable
global_m = None



# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global global_m
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        uploaded_file = request.files['file']  # Assuming the input field name is 'file'
        if uploaded_file.filename == '':
            return jsonify({'error': 'No file selected for uploading.'}), 400

        if uploaded_file:
            # Save the file to a temporary location
            temp_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(temp_path)

            logging.debug(f"File saved to {temp_path}")

            # Load the .mat file
            try:
                global_m = scipy.io.loadmat(temp_path, struct_as_record=True)
                logging.debug(f".mat file loaded successfully: {global_m}")
            except Exception as e:
                logging.error(f"Error loading .mat file: {e}")
                return jsonify({'error': f"Error loading .mat file: {e}"}), 500

            # Clean up: Delete the temporary file
            os.remove(temp_path)
            logging.debug(f"Temporary file {temp_path} deleted")

            return jsonify({'message': 'File uploaded and loaded successfully.'})
        else:
            return jsonify({'error': 'No file uploaded.'}), 400
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500
































































@app.route('/run_model', methods=['POST'])
def run_model():
    image_data = []
    # Plot PSD
    # plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']], chan_lab=['left', 'center', 'right'], maxy=1000)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    m=global_m
    # m = scipy.io.loadmat(r'C:\Users\samga\OneDrive\Desktop\newProject\BCICIV_calib_ds1d.mat', struct_as_record=True)
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]
    def psd(trials):    
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((nchannels, 101, ntrials))

        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()
                    
        return trials_PSD, freqs# Apply the function
    psd_r, freqs = psd(trials[cl1])
    psd_f, freqs = psd(trials[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
        plt.figure(figsize=(12,5))    
        nchans = len(chan_ind)
        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)   
        # Enumerate over the channels
        for i,ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows,ncols,i+1)
            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
            # All plot decoration below...
            plt.xlim(1,30)        
            if maxy != None:
                plt.ylim(0,maxy)
            else:
                plt.ylim(1,1200)    
            plt.grid()    
            plt.xlabel('Frequency (Hz)')        
            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])
            plt.legend()        
        plt.tight_layout()
    # plot_psd(
    # trials_PSD,
    # freqs,
    # [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    # chan_lab=['left', 'center', 'right'],
    # maxy=1000
    # )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    def bandpass(trials, lo, hi, sample_rate):
        a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
        
        return trials_filt
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    psd_r, freqs = psd(trials_filt[cl1])
    psd_f, freqs = psd(trials_filt[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    # plot_psd(trials_PSD,freqs,[channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],chan_lab=['left', 'center', 'right'],maxy=600 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    
    def logvar(trials):
        '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
        return np.log(np.var(trials, axis=1))


# In[13]:


# Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                    cl2: logvar(trials_filt[cl2])}


    # Below is a function to visualize the logvar of each channel as a bar chart:

    # In[14]:


    def plot_logvar(trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12,5))
        
        x0 = np.arange(nchannels)
        x1 = np.arange(nchannels) + 0.4

        y0 = np.mean(trials[cl1], axis=1)
        y1 = np.mean(trials[cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b')
        plt.bar(x1, y1, width=0.4, color='r')

        plt.xlim(-0.5, nchannels+0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend(cl_lab)


    # In[15]:


    # Plot the log-vars
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")


    # We see that most channels show a small difference in the log-var of the signal between the two classes. The next step is to go from 118 channels to only a few channel mixtures. The CSP algorithm calculates mixtures of channels that are designed to maximize the difference in variation between two classes. These mixures are called spatial filters.

    # In[16]:


    from numpy import linalg

    def cov(trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

    def csp(trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = cov(trials_r)
        cov_f = cov(trials_f)
        P = whitening(cov_r + cov_f)
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp


    # In[17]:


    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                cl2: apply_mix(W, trials_filt[cl2])}


    # To see the result of the CSP algorithm, we plot the log-var like we did before:

    # In[18]:


    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                    cl2: logvar(trials_csp[cl2])}
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Instead of 118 channels, we now have 118 mixtures of channels, called components. They are the result of 118 spatial filters applied to the data.
    # 
    # The first filters maximize the variation of the first class, while minimizing the variation of the second. The last filters maximize the variation of the second class, while minimizing the variation of the first.
    # 
    # This is also visible in a PSD plot. The code below plots the PSD for the first and last components as well as one in the middle:

    # In[19]:


    psd_r, freqs = psd(trials_csp[cl1])
    psd_f, freqs = psd(trials_csp[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}

    # plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # In order to see how well we can differentiate between the two classes, a scatter plot is a useful tool. Here both classes are plotted on a 2-dimensional plane: the x-axis is the first CSP component, the y-axis is the last.

    # In[20]:


    def plot_scatter(left, foot):
        plt.figure()
        plt.scatter(left[0,:], left[-1,:], color='b')
        plt.scatter(foot[0,:], foot[-1,:], color='r')
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend(cl_lab)


    # In[21]:


    # plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Percentage of trials to use for training (50-50 split here)
    train_percentage = 0.8

    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
    ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
    ntest_r = trials_filt[cl1].shape[2] - ntrain_r
    ntest_f = trials_filt[cl2].shape[2] - ntrain_f

    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
            cl2: trials_filt[cl2][:,:,:ntrain_f]}

    test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
            cl2: trials_filt[cl2][:,:,ntrain_f:]}

    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])

    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])

    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][comp,:,:]
    train[cl2] = train[cl2][comp,:,:]
    test[cl1] = test[cl1][comp,:,:]
    test[cl2] = test[cl2][comp,:,:]

    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])

    # Prepare the training and testing data for the SVM model
    X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T  # Combine and transpose
    y_train = np.concatenate((np.zeros(train[cl1].shape[1]), np.ones(train[cl2].shape[1])))

    X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T  # Combine and transpose
    y_test = np.concatenate((np.zeros(test[cl1].shape[1]), np.ones(test[cl2].shape[1])))

    # Train the SVM model
    svm_model = SVC(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.
    svm_model.fit(X_train, y_train)

    # Predict and evaluate the model on the test data
    y_pred = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'
    # Optional: Calculate and print other metrics (e.g., confusion matrix, precision, recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    conf_matrix_text = np.array2string(conf_matrix)
    classification_report_text = classification_report(y_test, y_pred)

    # Visualize the CSP-transformed data
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label=cl1, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label=cl2, alpha=0.6)
    plt.title('CSP-Transformed Training Data')
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.legend()
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Visualize the decision boundary
    def plot_decision_boundary(clf, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=cl1, alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=cl2, alpha=0.6)
        plt.title('SVM Decision Boundary')
        plt.xlabel('First Component')
        plt.ylabel('Last Component')
        plt.legend()

    plot_decision_boundary(svm_model, X_train, y_train)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    plot_decision_boundary(svm_model, X_test, y_test)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

        # # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cl1, cl2], yticklabels=[cl1, cl2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    # Combine all results
    results = {
        'accuracy': accuracy_text,
        'confusion_matrix': conf_matrix_text,
        'classification_report': classification_report_text,
        'images': image_data
    }
    return jsonify(results)
    # SVM END Here 
















@app.route('/KNN_model', methods=['POST'])
def KNN_model():
    image_data = []
    # Plot PSD
    # plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']], chan_lab=['left', 'center', 'right'], maxy=1000)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    m=global_m
    # m = scipy.io.loadmat(r'C:\Users\samga\OneDrive\Desktop\newProject\BCICIV_calib_ds1d.mat', struct_as_record=True)
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]
    def psd(trials):    
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((nchannels, 101, ntrials))

        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()
                    
        return trials_PSD, freqs# Apply the function
    psd_r, freqs = psd(trials[cl1])
    psd_f, freqs = psd(trials[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
        plt.figure(figsize=(12,5))    
        nchans = len(chan_ind)
        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)   
        # Enumerate over the channels
        for i,ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows,ncols,i+1)
            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
            # All plot decoration below...
            plt.xlim(1,30)        
            if maxy != None:
                plt.ylim(0,maxy)
            else:
                plt.ylim(1,1200)    
            plt.grid()    
            plt.xlabel('Frequency (Hz)')        
            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])
            plt.legend()        
        plt.tight_layout()
    # plot_psd(
    # trials_PSD,
    # freqs,
    # [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    # chan_lab=['left', 'center', 'right'],
    # maxy=1000
    # )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    def bandpass(trials, lo, hi, sample_rate):
        a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
        
        return trials_filt
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    psd_r, freqs = psd(trials_filt[cl1])
    psd_f, freqs = psd(trials_filt[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    # plot_psd(trials_PSD,freqs,[channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],chan_lab=['left', 'center', 'right'],maxy=600 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    
    def logvar(trials):
        '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
        return np.log(np.var(trials, axis=1))


# In[13]:


# Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                    cl2: logvar(trials_filt[cl2])}


    # Below is a function to visualize the logvar of each channel as a bar chart:

    # In[14]:


    def plot_logvar(trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12,5))
        
        x0 = np.arange(nchannels)
        x1 = np.arange(nchannels) + 0.4

        y0 = np.mean(trials[cl1], axis=1)
        y1 = np.mean(trials[cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b')
        plt.bar(x1, y1, width=0.4, color='r')

        plt.xlim(-0.5, nchannels+0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend(cl_lab)


    # In[15]:


    # Plot the log-vars
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")


    # We see that most channels show a small difference in the log-var of the signal between the two classes. The next step is to go from 118 channels to only a few channel mixtures. The CSP algorithm calculates mixtures of channels that are designed to maximize the difference in variation between two classes. These mixures are called spatial filters.

    # In[16]:


    from numpy import linalg

    def cov(trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

    def csp(trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = cov(trials_r)
        cov_f = cov(trials_f)
        P = whitening(cov_r + cov_f)
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp


    # In[17]:


    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                cl2: apply_mix(W, trials_filt[cl2])}


    # To see the result of the CSP algorithm, we plot the log-var like we did before:

    # In[18]:


    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                    cl2: logvar(trials_csp[cl2])}
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Instead of 118 channels, we now have 118 mixtures of channels, called components. They are the result of 118 spatial filters applied to the data.
    # 
    # The first filters maximize the variation of the first class, while minimizing the variation of the second. The last filters maximize the variation of the second class, while minimizing the variation of the first.
    # 
    # This is also visible in a PSD plot. The code below plots the PSD for the first and last components as well as one in the middle:

    # In[19]:


    psd_r, freqs = psd(trials_csp[cl1])
    psd_f, freqs = psd(trials_csp[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}

    # plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # In order to see how well we can differentiate between the two classes, a scatter plot is a useful tool. Here both classes are plotted on a 2-dimensional plane: the x-axis is the first CSP component, the y-axis is the last.

    # In[20]:


    def plot_scatter(left, foot):
        plt.figure()
        plt.scatter(left[0,:], left[-1,:], color='b')
        plt.scatter(foot[0,:], foot[-1,:], color='r')
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend(cl_lab)


    # In[21]:


    # plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Percentage of trials to use for training (50-50 split here)
    train_percentage = 0.8

    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
    ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
    ntest_r = trials_filt[cl1].shape[2] - ntrain_r
    ntest_f = trials_filt[cl2].shape[2] - ntrain_f

    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
            cl2: trials_filt[cl2][:,:,:ntrain_f]}

    test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
            cl2: trials_filt[cl2][:,:,ntrain_f:]}

    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])

    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])

    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][comp,:,:]
    train[cl2] = train[cl2][comp,:,:]
    test[cl1] = test[cl1][comp,:,:]
    test[cl2] = test[cl2][comp,:,:]

    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])

    # Prepare the training and testing data for the SVM model
    X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T  # Combine and transpose
    y_train = np.concatenate((np.zeros(train[cl1].shape[1]), np.ones(train[cl2].shape[1])))

    X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T  # Combine and transpose
    y_test = np.concatenate((np.zeros(test[cl1].shape[1]), np.ones(test[cl2].shape[1])))

    # Train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    knn_model.fit(X_train, y_train)

    # Predict and evaluate the model on  the test data
    y_pred = knn_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'
    # Optional: Calculate and print other metrics (e.g., confusion matrix, precision, recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    conf_matrix_text = np.array2string(conf_matrix)
    classification_report_text = classification_report(y_test, y_pred)

    # Visualize the CSP-transformed data
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label=cl1, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label=cl2, alpha=0.6)
    plt.title('CSP-Transformed Training Data')
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.legend()
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Visualize the decision boundary
    def plot_decision_boundary(clf, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=cl1, alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=cl2, alpha=0.6)
        plt.title('knn Decision Boundary')
        plt.xlabel('First Component')
        plt.ylabel('Last Component')
        plt.legend()

    plot_decision_boundary(knn_model, X_train, y_train)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    plot_decision_boundary(knn_model, X_test, y_test)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

        # # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cl1, cl2], yticklabels=[cl1, cl2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    # Combine all results
    results = {
        'accuracy': accuracy_text,
        'confusion_matrix': conf_matrix_text,
        'classification_report': classification_report_text,
        'images': image_data
    }
    return jsonify(results)
    # KNN END Here 
























@app.route('/naiveBayes', methods=['POST'])
def naiveBayes():
    image_data = []
    # Plot PSD
    # plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']], chan_lab=['left', 'center', 'right'], maxy=1000)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    m=global_m
    # m = scipy.io.loadmat(r'C:\Users\samga\OneDrive\Desktop\newProject\BCICIV_calib_ds1d.mat', struct_as_record=True)
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]
    def psd(trials):    
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((nchannels, 101, ntrials))

        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()
                    
        return trials_PSD, freqs# Apply the function
    psd_r, freqs = psd(trials[cl1])
    psd_f, freqs = psd(trials[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
        plt.figure(figsize=(12,5))    
        nchans = len(chan_ind)
        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)   
        # Enumerate over the channels
        for i,ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows,ncols,i+1)
            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
            # All plot decoration below...
            plt.xlim(1,30)        
            if maxy != None:
                plt.ylim(0,maxy)
            else:
                plt.ylim(1,1200)    
            plt.grid()    
            plt.xlabel('Frequency (Hz)')        
            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])
            plt.legend()        
        plt.tight_layout()
    # plot_psd(
    # trials_PSD,
    # freqs,
    # [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    # chan_lab=['left', 'center', 'right'],
    # maxy=1000
    # )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    def bandpass(trials, lo, hi, sample_rate):
        a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
        
        return trials_filt
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    psd_r, freqs = psd(trials_filt[cl1])
    psd_f, freqs = psd(trials_filt[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    # plot_psd(trials_PSD,freqs,[channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],chan_lab=['left', 'center', 'right'],maxy=600 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    
    def logvar(trials):
        '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
        return np.log(np.var(trials, axis=1))


# In[13]:


# Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                    cl2: logvar(trials_filt[cl2])}


    # Below is a function to visualize the logvar of each channel as a bar chart:

    # In[14]:


    def plot_logvar(trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12,5))
        
        x0 = np.arange(nchannels)
        x1 = np.arange(nchannels) + 0.4

        y0 = np.mean(trials[cl1], axis=1)
        y1 = np.mean(trials[cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b')
        plt.bar(x1, y1, width=0.4, color='r')

        plt.xlim(-0.5, nchannels+0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend(cl_lab)


    # In[15]:


    # Plot the log-vars
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")


    # We see that most channels show a small difference in the log-var of the signal between the two classes. The next step is to go from 118 channels to only a few channel mixtures. The CSP algorithm calculates mixtures of channels that are designed to maximize the difference in variation between two classes. These mixures are called spatial filters.

    # In[16]:


    from numpy import linalg

    def cov(trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

    def csp(trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = cov(trials_r)
        cov_f = cov(trials_f)
        P = whitening(cov_r + cov_f)
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp


    # In[17]:


    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                cl2: apply_mix(W, trials_filt[cl2])}


    # To see the result of the CSP algorithm, we plot the log-var like we did before:

    # In[18]:


    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                    cl2: logvar(trials_csp[cl2])}
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Instead of 118 channels, we now have 118 mixtures of channels, called components. They are the result of 118 spatial filters applied to the data.
    # 
    # The first filters maximize the variation of the first class, while minimizing the variation of the second. The last filters maximize the variation of the second class, while minimizing the variation of the first.
    # 
    # This is also visible in a PSD plot. The code below plots the PSD for the first and last components as well as one in the middle:

    # In[19]:


    psd_r, freqs = psd(trials_csp[cl1])
    psd_f, freqs = psd(trials_csp[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}

    # plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # In order to see how well we can differentiate between the two classes, a scatter plot is a useful tool. Here both classes are plotted on a 2-dimensional plane: the x-axis is the first CSP component, the y-axis is the last.

    # In[20]:


    def plot_scatter(left, foot):
        plt.figure()
        plt.scatter(left[0,:], left[-1,:], color='b')
        plt.scatter(foot[0,:], foot[-1,:], color='r')
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend(cl_lab)


    # In[21]:


    # plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Percentage of trials to use for training (50-50 split here)
    train_percentage = 0.8

    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
    ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
    ntest_r = trials_filt[cl1].shape[2] - ntrain_r
    ntest_f = trials_filt[cl2].shape[2] - ntrain_f

    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
            cl2: trials_filt[cl2][:,:,:ntrain_f]}

    test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
            cl2: trials_filt[cl2][:,:,ntrain_f:]}

    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])

    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])

    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][comp,:,:]
    train[cl2] = train[cl2][comp,:,:]
    test[cl1] = test[cl1][comp,:,:]
    test[cl2] = test[cl2][comp,:,:]

    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])

    # Prepare the training and testing data for the SVM model
    X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T  # Combine and transpose
    y_train = np.concatenate((np.zeros(train[cl1].shape[1]), np.ones(train[cl2].shape[1])))

    X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T  # Combine and transpose
    y_test = np.concatenate((np.zeros(test[cl1].shape[1]), np.ones(test[cl2].shape[1])))

    # Train the Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predict and evaluate the model on the test data
    y_pred = nb_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'
    # Optional: Calculate and print other metrics (e.g., confusion matrix, precision, recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    conf_matrix_text = np.array2string(conf_matrix)
    classification_report_text = classification_report(y_test, y_pred)

    # Visualize the CSP-transformed data
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label=cl1, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label=cl2, alpha=0.6)
    plt.title('CSP-Transformed Training Data')
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.legend()
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Visualize the decision boundary
    def plot_decision_boundary(clf, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=cl1, alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=cl2, alpha=0.6)
        plt.title('navie byers Decision Boundary')
        plt.xlabel('First Component')
        plt.ylabel('Last Component')
        plt.legend()

    plot_decision_boundary(nb_model, X_train, y_train)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    plot_decision_boundary(nb_model, X_test, y_test)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

        # # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cl1, cl2], yticklabels=[cl1, cl2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    # Combine all results
    results = {
        'accuracy': accuracy_text,
        'confusion_matrix': conf_matrix_text,
        'classification_report': classification_report_text,
        'images': image_data
    }
    return jsonify(results)

# NAVIE BYERS END










@app.route('/xgboost', methods=['POST'])
def xgboost():
    image_data = []
    # Plot PSD
    # plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']], chan_lab=['left', 'center', 'right'], maxy=1000)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    m=global_m
    # m = scipy.io.loadmat(r'C:\Users\samga\OneDrive\Desktop\newProject\BCICIV_calib_ds1d.mat', struct_as_record=True)
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]
    def psd(trials):    
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((nchannels, 101, ntrials))

        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()
                    
        return trials_PSD, freqs# Apply the function
    psd_r, freqs = psd(trials[cl1])
    psd_f, freqs = psd(trials[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
        plt.figure(figsize=(12,5))    
        nchans = len(chan_ind)
        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)   
        # Enumerate over the channels
        for i,ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows,ncols,i+1)
            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
            # All plot decoration below...
            plt.xlim(1,30)        
            if maxy != None:
                plt.ylim(0,maxy)
            else:
                plt.ylim(1,1200)    
            plt.grid()    
            plt.xlabel('Frequency (Hz)')        
            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])
            plt.legend()        
        plt.tight_layout()
    # plot_psd(
    # trials_PSD,
    # freqs,
    # [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    # chan_lab=['left', 'center', 'right'],
    # maxy=1000
    # )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    def bandpass(trials, lo, hi, sample_rate):
        a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
        
        return trials_filt
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    psd_r, freqs = psd(trials_filt[cl1])
    psd_f, freqs = psd(trials_filt[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    # plot_psd(trials_PSD,freqs,[channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],chan_lab=['left', 'center', 'right'],maxy=600 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    
    def logvar(trials):
        '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
        return np.log(np.var(trials, axis=1))


# In[13]:


# Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                    cl2: logvar(trials_filt[cl2])}


    # Below is a function to visualize the logvar of each channel as a bar chart:

    # In[14]:


    def plot_logvar(trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12,5))
        
        x0 = np.arange(nchannels)
        x1 = np.arange(nchannels) + 0.4

        y0 = np.mean(trials[cl1], axis=1)
        y1 = np.mean(trials[cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b')
        plt.bar(x1, y1, width=0.4, color='r')

        plt.xlim(-0.5, nchannels+0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend(cl_lab)


    # In[15]:


    # Plot the log-vars
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")


    # We see that most channels show a small difference in the log-var of the signal between the two classes. The next step is to go from 118 channels to only a few channel mixtures. The CSP algorithm calculates mixtures of channels that are designed to maximize the difference in variation between two classes. These mixures are called spatial filters.

    # In[16]:


    from numpy import linalg

    def cov(trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

    def csp(trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = cov(trials_r)
        cov_f = cov(trials_f)
        P = whitening(cov_r + cov_f)
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp


    # In[17]:


    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                cl2: apply_mix(W, trials_filt[cl2])}


    # To see the result of the CSP algorithm, we plot the log-var like we did before:

    # In[18]:


    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                    cl2: logvar(trials_csp[cl2])}
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Instead of 118 channels, we now have 118 mixtures of channels, called components. They are the result of 118 spatial filters applied to the data.
    # 
    # The first filters maximize the variation of the first class, while minimizing the variation of the second. The last filters maximize the variation of the second class, while minimizing the variation of the first.
    # 
    # This is also visible in a PSD plot. The code below plots the PSD for the first and last components as well as one in the middle:

    # In[19]:


    psd_r, freqs = psd(trials_csp[cl1])
    psd_f, freqs = psd(trials_csp[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}

    # plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # In order to see how well we can differentiate between the two classes, a scatter plot is a useful tool. Here both classes are plotted on a 2-dimensional plane: the x-axis is the first CSP component, the y-axis is the last.

    # In[20]:


    def plot_scatter(left, foot):
        plt.figure()
        plt.scatter(left[0,:], left[-1,:], color='b')
        plt.scatter(foot[0,:], foot[-1,:], color='r')
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend(cl_lab)


    # In[21]:


    # plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Percentage of trials to use for training (50-50 split here)
    train_percentage = 0.8

    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
    ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
    ntest_r = trials_filt[cl1].shape[2] - ntrain_r
    ntest_f = trials_filt[cl2].shape[2] - ntrain_f

    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
            cl2: trials_filt[cl2][:,:,:ntrain_f]}

    test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
            cl2: trials_filt[cl2][:,:,ntrain_f:]}

    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])

    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])

    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][comp,:,:]
    train[cl2] = train[cl2][comp,:,:]
    test[cl1] = test[cl1][comp,:,:]
    test[cl2] = test[cl2][comp,:,:]

    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])

    # Prepare the training and testing data for the SVM model
    X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T  # Combine and transpose
    y_train = np.concatenate((np.zeros(train[cl1].shape[1]), np.ones(train[cl2].shape[1])))

    X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T  # Combine and transpose
    y_test = np.concatenate((np.zeros(test[cl1].shape[1]), np.ones(test[cl2].shape[1])))

    # Train the XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Predict and evaluate the model on the test data
    y_pred = xgb_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'
    # Optional: Calculate and print other metrics (e.g., confusion matrix, precision, recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    conf_matrix_text = np.array2string(conf_matrix)
    classification_report_text = classification_report(y_test, y_pred)

    # Visualize the CSP-transformed data
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label=cl1, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label=cl2, alpha=0.6)
    plt.title('CSP-Transformed Training Data')
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.legend()
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Visualize the decision boundary
    def plot_decision_boundary(clf, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=cl1, alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=cl2, alpha=0.6)
        plt.title('XGBoost Decision Boundary')
        plt.xlabel('First Component')
        plt.ylabel('Last Component')
        plt.legend()

    plot_decision_boundary(xgb_model, X_train, y_train)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    plot_decision_boundary(xgb_model, X_test, y_test)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

        # # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cl1, cl2], yticklabels=[cl1, cl2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    # Combine all results
    results = {
        'accuracy': accuracy_text,
        'confusion_matrix': conf_matrix_text,
        'classification_report': classification_report_text,
        'images': image_data
    }
    return jsonify(results)
    # XGBoost END Here 
    
    
    
    






























    
    
@app.route('/logisticRegression', methods=['POST'])
def logisticRegression():
    image_data = []
    # Plot PSD
    # plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']], chan_lab=['left', 'center', 'right'], maxy=1000)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    m=global_m
    # m = scipy.io.loadmat(r'C:\Users\samga\OneDrive\Desktop\newProject\BCICIV_calib_ds1d.mat', struct_as_record=True)
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]
    def psd(trials):    
        ntrials = trials.shape[2]
        trials_PSD = np.zeros((nchannels, 101, ntrials))

        # Iterate over trials and channels
        for trial in range(ntrials):
            for ch in range(nchannels):
                # Calculate the PSD
                (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
                trials_PSD[ch, :, trial] = PSD.ravel()
                    
        return trials_PSD, freqs# Apply the function
    psd_r, freqs = psd(trials[cl1])
    psd_f, freqs = psd(trials[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
        plt.figure(figsize=(12,5))    
        nchans = len(chan_ind)
        # Maximum of 3 plots per row
        nrows = int(np.ceil(nchans / 3))
        ncols = min(3, nchans)   
        # Enumerate over the channels
        for i,ch in enumerate(chan_ind):
            # Figure out which subplot to draw to
            plt.subplot(nrows,ncols,i+1)
            # Plot the PSD for each class
            for cl in trials.keys():
                plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
            # All plot decoration below...
            plt.xlim(1,30)        
            if maxy != None:
                plt.ylim(0,maxy)
            else:
                plt.ylim(1,1200)    
            plt.grid()    
            plt.xlabel('Frequency (Hz)')        
            if chan_lab == None:
                plt.title('Channel %d' % (ch+1))
            else:
                plt.title(chan_lab[i])
            plt.legend()        
        plt.tight_layout()
    # plot_psd(
    # trials_PSD,
    # freqs,
    # [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    # chan_lab=['left', 'center', 'right'],
    # maxy=1000
    # # )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    def bandpass(trials, lo, hi, sample_rate):
        a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

        # Applying the filter to each trial
        ntrials = trials.shape[2]
        trials_filt = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
        
        return trials_filt
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    psd_r, freqs = psd(trials_filt[cl1])
    psd_f, freqs = psd(trials_filt[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}
    # plot_psd(trials_PSD,freqs,[channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],chan_lab=['left', 'center', 'right'],maxy=600 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    
    def logvar(trials):
        '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
        return np.log(np.var(trials, axis=1))


# In[13]:


# Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                    cl2: logvar(trials_filt[cl2])}


    # Below is a function to visualize the logvar of each channel as a bar chart:

    # In[14]:


    def plot_logvar(trials):
        '''
        Plots the log-var of each channel/component.
        arguments:
            trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
        '''
        plt.figure(figsize=(12,5))
        
        x0 = np.arange(nchannels)
        x1 = np.arange(nchannels) + 0.4

        y0 = np.mean(trials[cl1], axis=1)
        y1 = np.mean(trials[cl2], axis=1)

        plt.bar(x0, y0, width=0.5, color='b')
        plt.bar(x1, y1, width=0.4, color='r')

        plt.xlim(-0.5, nchannels+0.5)

        plt.gca().yaxis.grid(True)
        plt.title('log-var of each channel/component')
        plt.xlabel('channels/components')
        plt.ylabel('log-var')
        plt.legend(cl_lab)


    # In[15]:


    # Plot the log-vars
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")


    # We see that most channels show a small difference in the log-var of the signal between the two classes. The next step is to go from 118 channels to only a few channel mixtures. The CSP algorithm calculates mixtures of channels that are designed to maximize the difference in variation between two classes. These mixures are called spatial filters.

    # In[16]:


    from numpy import linalg

    def cov(trials):
        ''' Calculate the covariance for each trial and return their average '''
        ntrials = trials.shape[2]
        covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
        return np.mean(covs, axis=0)

    def whitening(sigma):
        ''' Calculate a whitening matrix for covariance matrix sigma. '''
        U, l, _ = linalg.svd(sigma)
        return U.dot( np.diag(l ** -0.5) )

    def csp(trials_r, trials_f):
        '''
        Calculate the CSP transformation matrix W.
        arguments:
            trials_r - Array (channels x samples x trials) containing right hand movement trials
            trials_f - Array (channels x samples x trials) containing foot movement trials
        returns:
            Mixing matrix W
        '''
        cov_r = cov(trials_r)
        cov_f = cov(trials_f)
        P = whitening(cov_r + cov_f)
        B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
        W = P.dot(B)
        return W

    def apply_mix(W, trials):
        ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
        ntrials = trials.shape[2]
        trials_csp = np.zeros((nchannels, nsamples, ntrials))
        for i in range(ntrials):
            trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        return trials_csp


    # In[17]:


    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                cl2: apply_mix(W, trials_filt[cl2])}


    # To see the result of the CSP algorithm, we plot the log-var like we did before:

    # In[18]:


    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                    cl2: logvar(trials_csp[cl2])}
    # plot_logvar(trials_logvar)
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Instead of 118 channels, we now have 118 mixtures of channels, called components. They are the result of 118 spatial filters applied to the data.
    # 
    # The first filters maximize the variation of the first class, while minimizing the variation of the second. The last filters maximize the variation of the second class, while minimizing the variation of the first.
    # 
    # This is also visible in a PSD plot. The code below plots the PSD for the first and last components as well as one in the middle:

    # In[19]:


    psd_r, freqs = psd(trials_csp[cl1])
    psd_f, freqs = psd(trials_csp[cl2])
    trials_PSD = {cl1: psd_r, cl2: psd_f}

    # plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # In order to see how well we can differentiate between the two classes, a scatter plot is a useful tool. Here both classes are plotted on a 2-dimensional plane: the x-axis is the first CSP component, the y-axis is the last.

    # In[20]:


    def plot_scatter(left, foot):
        plt.figure()
        plt.scatter(left[0,:], left[-1,:], color='b')
        plt.scatter(foot[0,:], foot[-1,:], color='r')
        plt.xlabel('Last component')
        plt.ylabel('First component')
        plt.legend(cl_lab)


    # In[21]:


    # plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
    # image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Percentage of trials to use for training (50-50 split here)
    train_percentage = 0.8

    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
    ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
    ntest_r = trials_filt[cl1].shape[2] - ntrain_r
    ntest_f = trials_filt[cl2].shape[2] - ntrain_f

    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
            cl2: trials_filt[cl2][:,:,:ntrain_f]}

    test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
            cl2: trials_filt[cl2][:,:,ntrain_f:]}

    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])

    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])

    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][comp,:,:]
    train[cl2] = train[cl2][comp,:,:]
    test[cl1] = test[cl1][comp,:,:]
    test[cl2] = test[cl2][comp,:,:]

    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])

    # Prepare the training and testing data for the SVM model
    X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T  # Combine and transpose
    y_train = np.concatenate((np.zeros(train[cl1].shape[1]), np.ones(train[cl2].shape[1])))

    X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T  # Combine and transpose
    y_test = np.concatenate((np.zeros(test[cl1].shape[1]), np.ones(test[cl2].shape[1])))

    # Train the Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Predict and evaluate the model on the test data
    y_pred = lr_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'
    # Optional: Calculate and print other metrics (e.g., confusion matrix, precision, recall)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    # print(classification_report(y_test, y_pred))
    conf_matrix_text = np.array2string(conf_matrix)
    classification_report_text = classification_report(y_test, y_pred)

    # Visualize the CSP-transformed data
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label=cl1, alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label=cl2, alpha=0.6)
    plt.title('CSP-Transformed Training Data')
    plt.xlabel('First Component')
    plt.ylabel('Last Component')
    plt.legend()
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

    # Visualize the decision boundary
    def plot_decision_boundary(clf, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=cl1, alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=cl2, alpha=0.6)
        plt.title('Logistic Regression Decision Boundary')
        plt.xlabel('First Component')
        plt.ylabel('Last Component')
        plt.legend()

    plot_decision_boundary(lr_model, X_train, y_train)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    plot_decision_boundary(lr_model, X_test, y_test)
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")

        # # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cl1, cl2], yticklabels=[cl1, cl2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    image_data.append(f"data:image/png;base64,{generate_base64_image(plt)}")
    
    # Combine all results
    results = {
        'accuracy': accuracy_text,
        'confusion_matrix': conf_matrix_text,
        'classification_report': classification_report_text,
        'images': image_data
    }
    return jsonify(results)
    # LogisticRegression END HERE











    
    
    
    
    
    
    










if __name__ == '__main__':
    app.run(debug=True)
