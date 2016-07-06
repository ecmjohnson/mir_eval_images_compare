from __future__ import print_function
import numpy as np
import matlab_wrapper
import soundfile as sf
import os
import mir_eval
import argparse
import time


class matlabwrapper(matlab_wrapper.MatlabSession):
    """A simple wrapper for matlab_wrapper to load matlab only once"""
    def eval_images(self, estimates, sources, rate):
        self.put('es', estimates)
        self.put('s', sources)
        self.put('fs', rate)
        self.eval(
            '[SDR,ISR,SIR,SAR] = bss_eval_images(es, s)'
        )
        SDR = self.get('SDR')
        ISR = self.get('ISR')
        SIR = self.get('SIR')
        SAR = self.get('SAR')

        return SDR, ISR, SIR, SAR

    def sisec_eval_images(self, estimates, sources, rate):
        self.put('es', estimates)
        self.put('s', sources)
        self.put('fs', rate)
        self.eval(
            '[SDR,ISR,SIR,SAR] = sisec_eval_images(es, s)'
        )
        SDR = self.get('SDR')
        ISR = self.get('ISR')
        SIR = self.get('SIR')
        SAR = self.get('SAR')

        return SDR, ISR, SIR, SAR

    def eval_sources(self, estimates, sources, rate):
        self.put('es', estimates)
        self.put('s', sources)
        self.put('fs', rate)
        self.eval(
            '[SDR,SIR,SAR] = bss_eval_sources(es, s)'
        )
        SDR = self.get('SDR')
        SIR = self.get('SIR')
        SAR = self.get('SAR')

        return SDR, np.nan, SIR, SAR

    def start(self):
        matlab_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'matlab'
        )
        self.eval('addpath(\'%s/\')' % matlab_path)


class BSSeval(object):
    def __init__(self, methods):

        # list of methods
        self.methods = methods

        # start matlab
        if ("bss_eval_matlab_images" in methods
                or "bss_eval_matlab_sources" in methods):
            self.matlab = matlabwrapper(options="-nosplash -nodesktop -nojvm")
            self.matlab.start()
        else:
            self.matlab = None

    def evaluate(self, estimates, sources, rate, verbose=True):
        """Universal BSS evaluate frontend for several evaluators

        Parameters
        ----------
        sources : np.ndarray, shape=(nsrc, nsampl, nchan)
            array containing true reference sources
        estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
            array containing estimated sources

        Returns
        -------
        SDR : np.ndarray, shape=(nsrc,)
            vector of Signal to Distortion Ratios (SDR)
        ISR : np.ndarray, shape=(nsrc,)
            vector of Source to Spatial Distortion Image (ISR)
        SIR : np.ndarray, shape=(nsrc,)
            vector of Source to Interference Ratios (SIR)
        SAR : np.ndarray, shape=(nsrc,)
            vector of Sources to Artifacts Ratios (SAR)
        """

        results = []
        for method in self.methods:

            print('Beginning evaluation of method: ' + method)

            if method == "mir_eval_sources":

                # convert the stereo audio into mono
                src_mono = np.mean(np.array(sources), axis=-1)
                est_mono = np.mean(np.array(estimates), axis=-1)

                start = time.time()
                SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(
                    src_mono,
                    est_mono
                )
                end = time.time()
                duration = end - start
                timings[method] = duration

                results.append(
                    {
                        'Method': method,
                        'SDR': SDR,
                        'SIR': SIR,
                        'ISR': np.nan,
                        'SAR': SAR,
                    }
                )

            if method == "mir_eval_images":

                start = time.time()
                SDR, ISR, SIR, SAR, perm = mir_eval.separation.bss_eval_images(
                    sources,
                    estimates
                )
                end = time.time()
                duration = end - start
                timings[method] = duration

                results.append(
                    {
                        'Method': method,
                        'SDR': SDR,
                        'SIR': SIR,
                        'ISR': ISR,
                        'SAR': SAR,
                    }
                )

            if method == "mir_eval_images_noperm":

                start = time.time()
                SDR, ISR, SIR, SAR, perm = mir_eval.separation.bss_eval_images(
                    sources,
                    estimates,
                    False
                )
                end = time.time()
                duration = end - start
                timings[method] = duration

                results.append(
                    {
                        'Method': method,
                        'SDR': SDR,
                        'SIR': SIR,
                        'ISR': ISR,
                        'SAR': SAR,
                    }
                )

            if method == "bss_eval_matlab_images":

                start = time.time()
                SDR, ISR, SIR, SAR = self.matlab.eval_images(
                    estimates,
                    sources,
                    rate
                )
                end = time.time()
                duration = end - start
                timings[method] = duration

                results.append(
                    {
                        'Method': method,
                        'SDR': SDR,
                        'SIR': SIR,
                        'ISR': ISR,
                        'SAR': SAR,
                    }
                )

            if method == "bss_eval_matlab_sources":

                # convert the stereo audio into mono
                src_mono = np.mean(np.array(sources), axis=-1)
                est_mono = np.mean(np.array(estimates), axis=-1)

                start = time.time()
                SDR, ISR, SIR, SAR = self.matlab.eval_sources(
                    est_mono,
                    src_mono,
                    rate
                )
                end = time.time()
                duration = end - start
                timings[method] = duration

                results.append(
                    {
                        'Method': method,
                        'SDR': SDR,
                        'SIR': SIR,
                        'ISR': ISR,
                        'SAR': SAR,
                    }
                )

        return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ALABS Python Source Separation Evaluation Toolkit')

    parser.add_argument('src', type=str, default=None, nargs='?',
                        help='Target Source')

    parser.add_argument('est', type=str, default=None, nargs='?',
                        help='Estimated Source')

    args = parser.parse_args()

    if args.src and args.est is not None:
        # args.src and args.est are directories containing the files for each source
        est_dir = args.est.replace('\\', '') # format for os.listdir()
        est_contents = os.listdir(est_dir) # get the contents of the estimates folder
        est_files = [k for k in est_contents if '.wav' in k] # filter non-.wavs (just in case)
        #*** Assuming that the files in source and estimated dirs are named the same ***
        srcs_audio = None
        for source in est_files:
            # Read files into shape (nb_samples x nb_channels)
            src_audio, src_rate = sf.read(args.src + '/' + source)
            est_audio, est_rate = sf.read(args.est + '/' + source)
            # Handle unequal nb_samples ***should this even be allowed?***
            src_samples = len(src_audio); est_samples = len(est_audio);
            if (src_samples != est_samples):
                if (src_samples < est_samples):
                    src_audio = np.vstack((src_audio, np.zeros(((est_samples - src_samples), 2))))
                else:
                    est_audio = np.vstack((est_audio, np.zeros(((src_samples - est_samples), 2))))
            # Transpose to 3d for stacking
            src_audio = np.atleast_3d(src_audio).reshape((1, -1, 2))
            est_audio = np.atleast_3d(est_audio).reshape((1, -1, 2))
            # Stack into the source combinations
            if srcs_audio is None:
                srcs_audio = src_audio
                ests_audio = est_audio
            else:
                srcs_audio = np.vstack((srcs_audio, src_audio))
                ests_audio = np.vstack((ests_audio, est_audio))
        # Maintain compatibility
        src_audio = srcs_audio; est_audio = ests_audio;
    else:
        # just use some noise for testing
        src_audio = np.random.random((4, 45*44100, 2))
        src_rate = 44100
        est_audio = np.random.random((4, 45*44100, 2))
        est_rate = src_rate

    # Prepare timing dictionary
    timings = {}

    # Receive results
    bss = BSSeval([
        'mir_eval_images',
        'mir_eval_images_noperm',
        'bss_eval_matlab_images',
        'mir_eval_sources',
        'bss_eval_matlab_sources',
    ])
    results = bss.evaluate(est_audio, src_audio, src_rate)

    # Try for a readable output
    try:
        import texttable as txtbl
        table = txtbl.Texttable()
        table.set_cols_dtype(['t',  # method used
                              'f',  # SDR
                              'f',  # SIR
                              'f',  # ISR
                              'f']) # SAR
        table.set_cols_align(["c", "c", "c", "c", "c"])
        table.header(["Method", "SDR", "SIR", "ISR", "SAR"])
        for result in results:
            method = result['Method']
            try:
                sources = len(result['SDR'])
            except:
                sources = 1
            for i in range(sources):
                row = []
                if(i == 0):
                    row.append(method)
                else:
                    row.append('-')
                try:
                    row.append(result['SDR'][i])
                except:
                    row.append(result['SDR'])
                try:
                    row.append(result['SIR'][i])
                except:
                    row.append(result['SIR'])
                try:
                    row.append(result['ISR'][i])
                except:
                    try:
                        row.append(float(result['ISR']))
                    except ValueError:
                        row.append(float('NaN'))
                try:
                    row.append(result['SAR'][i])
                except:
                    row.append(result['SAR'])
                table.add_row(row)
        print(table.draw())

    except ImportError:
        # sad...
        print(result)

    print(timings)
    print('This gives the time to evaluate the methods')
    print('Be aware that they are not necessarily comparable (e.g. computing'
          ' permutations)!')
