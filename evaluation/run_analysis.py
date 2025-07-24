import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_analysis import ablation_study, feature_correlation_analysis, pca_analysis
import pickle as cPickle
import os

def load_data(dataFile):
    """Load the dataset"""
    with open(dataFile, 'rb') as f:
        Xd = cPickle.load(f, encoding='latin1')
    return Xd

def prepare_data(Xd):
    """Prepare data for analysis"""
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    # Split data
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = n_examples * 0.6
    train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Convert labels to one-hot
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    
    return X_train, X_test, Y_train, Y_test, lbl, train_idx, test_idx, mods, snrs

def plot_ablation_results(ablation_results, snrs):
    """Plot ablation study results"""
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy curves
    plt.plot(snrs, [ablation_results[snr]['CNN'] for snr in snrs], 
             marker='o', label='CNN Only', color='blue')
    plt.plot(snrs, [ablation_results[snr]['Features'] for snr in snrs], 
             marker='s', label='Features Only', color='red')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('Ablation Study Results')
    plt.grid(True)
    plt.legend()
    plt.savefig('ablation_results.png')
    
    # Calculate and print performance improvements
    improvements = {}
    for snr in snrs:
        cnn_acc = ablation_results[snr]['CNN']
        feature_acc = ablation_results[snr]['Features']
        improvements[snr] = {
            'Feature_Improvement': feature_acc - cnn_acc,
            'Percentage_Improvement': ((feature_acc - cnn_acc) / cnn_acc) * 100
        }
    
    return improvements

def main():
    # Load and prepare data
    dataFile = "data/RML2016.10a_dict.pkl"
    Xd = load_data(dataFile)
    X_train, X_test, Y_train, Y_test, lbl, train_idx, test_idx, mods, snrs = prepare_data(Xd)
    
    # Load artificial features
    filename = "data/A_P_data.pickle"
    with open(filename, 'rb') as file:
        M = cPickle.load(file)
    
    # Prepare artificial features (similar to AFECNN.py)
    # ... (copy the feature preparation code from AFECNN.py)
    
    # Run analyses
    ablation_results = ablation_study(X_train, X_test, extraData_train, extraData_test, 
                                    Y_train, Y_test, mods, snrs)
    
    corr_matrix, mod_correlations = feature_correlation_analysis(extraData_train, mods, 
                                                              train_idx, lbl)
    
    pca, X_pca = pca_analysis(extraData_train, mods, train_idx, lbl)
    
    # Plot and save results
    improvements = plot_ablation_results(ablation_results, snrs)
    
    # Save numerical results
    results = {
        'ablation_results': ablation_results,
        'improvements': improvements,
        'corr_matrix': corr_matrix,
        'mod_correlations': mod_correlations,
        'pca_results': {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }
    }
    
    np.save('analysis_results.npy', results)
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print("----------------")
    print("\nFeature Contribution:")
    for snr in snrs:
        print(f"SNR {snr}dB: {improvements[snr]['Percentage_Improvement']:.2f}% improvement")
    
    print("\nTop 3 Most Important Features (based on PCA):")
    feature_names = ['Param_R', 'M_1_real', 'M_1_imag', 'M_2', 'M_3_real', 'M_3_imag', 
                    'M_4', 'M_5', 'M_6', 'C_60_real', 'C_60_imag']
    for i in range(3):
        print(f"Component {i+1}: {feature_names[np.argmax(np.abs(pca.components_[i]))]}")

if __name__ == "__main__":
    main() 