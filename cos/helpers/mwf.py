import argparse
import itertools
import json
import multiprocessing.dummy as mp
import os

from pathlib import Path

import librosa
import numpy as np
import tqdm

from scipy.signal import stft, istft

from cos.helpers.eval_utils import compute_sdr
from cos.inference.evaluate_synthetic import get_items
from cos.helpers.utils import check_valid_dir


def invert(M,eps):
    """"inverting matrices M (matrices are the two last dimensions).
    This is assuming that these are 2x2 matrices, using the explicit
    inversion formula available in that case."""
    invDet = 1.0/(eps +  M[...,0,0]*M[...,1,1] - M[...,0,1]*M[...,1,0])
    invM = np.zeros(M.shape,dtype='complex')
    invM[...,0,0] =  invDet*M[...,1,1]
    invM[...,1,0] = -invDet*M[...,1,0]
    invM[...,0,1] = -invDet*M[...,0,1]
    invM[...,1,1] =  invDet*M[...,0,0]
    return invM


def compute_mwf(gt, mix):
    """
    Computes the Ideal Ratio Mask SI-SDR
    gt: (n_voices, n_channels, t)
    mix: (n_channels, t)
    """
    n_voices = gt.shape[0]
    nfft = 2048
    hop = 1024
    eps = np.finfo(np.float).eps

    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape # (6, nfft//2 +1, n_frame)

    # Allocate variables P: PSD, R: Spatial Covarianc Matrices
    P = []
    R = []
    for gt_idx in range(n_voices):
        # compute STFT of target source
        Yj = stft(gt[gt_idx], nperseg=nfft)[2]

        # Learn Power Spectral Density and spatial covariance matrix
        #-----------------------------------------------------------

        # 1/ compute observed covariance for source
        Rjj = np.zeros((F,T,I,I), dtype='complex')
        for (i1,i2) in itertools.product(range(I),range(I)):
            Rjj[...,i1,i2] = Yj[i1,...]*np.conj(Yj[i2,...])

        # 2/ compute first naive estimate of the source spectrogram as the
        #    average of spectrogram over channels
        P.append(np.mean(np.abs(Yj)**2,axis=0))

        # 3/ take the spatial covariance matrix as the average of
        #    the observed Rjj weighted Rjj by 1/Pj. This is because the
        #    covariance is modeled as Pj Rj
        R.append(np.mean(Rjj / (eps+P[-1][...,None,None]), axis = 1))

        # add some regularization to this estimate: normalize and add small
        # identify matrix, so we are sure it behaves well numerically.
        R[-1] = R[-1] * I/ np.trace(R[-1]) + eps * np.tile(np.eye(I,dtype='complex64')[None,...],(F,1,1))

        # 4/ Now refine the power spectral density estimate. This is to better
        #    estimate the PSD in case the source has some correlations between
        #    channels.

        #    invert Rj
        Rj_inv = invert(R[-1],eps)

        #    now compute the PSD
        P[-1]=0
        for (i1,i2) in itertools.product(range(I),range(I)):
            P[-1] +=  1./I*np.real(Rj_inv[:,i1,i2][:,None]*Rjj[...,i2,i1])
        
    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    Cxx = 0
    for gt_idx in range(n_voices):
        Cxx += P[gt_idx][...,None,None]*R[gt_idx][:,None,...]
    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx,eps)

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # computes multichannel Wiener gain as Pj Rj invCxx
        G = np.zeros(invCxx.shape,dtype='complex64')
        SR = P[gt_idx][...,None,None]*R[gt_idx][:,None,...]
        for (i1,i2,i3) in itertools.product(range(I),range(I),range(I)):
            G[...,i1,i2] += SR[...,i1,i3]*invCxx[...,i3,i2]
        SR = 0 #free memory

        # separates by (matrix-)multiplying this gain with the mix.
        Yj=0
        for i in range(I):
            Yj+=G[...,i]*X[i,...,None]
        Yj = np.rollaxis(Yj,-1) #gets channels back in first position

        # inverte to time domain
        target_estimate = istft(Yj)[1][:,:N]

        estimates.append(target_estimate)

    estimates = np.array(estimates) # (nvoice, 6, 6*sr)
    # eval
    eval_mix = np.repeat(mix[np.newaxis, :, :], n_voices, axis=0) # (nvoice, 6, 6*sr)
    eval_gt = gt # (nvoice, 6, 6*sr)
    eval_est = estimates

    SDR_in = []
    SDR_out = []
    for i in range(n_voices):
        SDR_in.append(compute_sdr(eval_gt[i], eval_mix[i], single_channel=False)) # scalar
        SDR_out.append(compute_sdr(eval_gt[i], eval_est[i], single_channel=False)) # scalar

    output = np.array([SDR_in, SDR_out]) # (2, nvoice)
    return output


def main(args):
    all_dirs = sorted(list(Path(args.input_dir).glob('[0-9]*')))
    all_dirs = [x for x in all_dirs if check_valid_dir(x, args.n_voices)]

    all_input_sdr = [0] * len(all_dirs)
    all_output_sdr = [0] * len(all_dirs)

    def evaluate_dir(idx):
        curr_dir = all_dirs[idx]
        # Loads the data
        mixed_data, gt = get_items(curr_dir, args)
        gt = np.array([x.data for x in gt])
        output = compute_mwf(gt, mixed_data)
        all_input_sdr[idx] = output[0]
        all_output_sdr[idx] = output[1]

    pool = mp.Pool(args.n_workers)
    with tqdm.tqdm(total=len(all_dirs)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(evaluate_dir, range(len(all_dirs)))):
            pbar.update()
    
    # tqdm.tqdm(pool.imap(evaluate_dir, range(len(all_dirs))), total=len(all_dirs))
    pool.close()
    pool.join()

    print("Median SI-SDRi: ",
          np.median(np.array(all_output_sdr).flatten() - np.array(all_input_sdr).flatten()))

    np.save("MWF_{}voices_{}kHz.npy".format(args.n_voices, args.sr),
            np.array([np.array(all_input_sdr).flatten(), np.array(all_output_sdr).flatten()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Path to the input dir")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels',
                        type=int,
                        default=2,
                        help="Number of channels")
    parser.add_argument('--n_workers',
                        type=int,
                        default=8,
                        help="Number of parallel workers")
    parser.add_argument('--n_voices',
                        type=int,
                        default=2,
                        help="Number of voices in the dataset")
    parser.add_argument('--alpha',
                        type=int,
                        default=1,
                        help="See the original SigSep code for an explanation")
    args = parser.parse_args()

    main(args)
