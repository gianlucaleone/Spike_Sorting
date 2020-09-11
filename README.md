# Spike_Sorting

This software pipeline permits to run FIR-based filtering, median-based spike detection, FSDE-based feature extraction and k-means clustering 
on single- and multiple-channel neural data recordings.

### Prerequisites

We suggest the use of a Python 3.6 environment on Anaconda.
Provide some neural data. Add instructions to read neural data if needed. 
The code is provided with a .mat-loading routine.

## Running the tests

Go into the project folder (ss).

run in Python:

import scripts.SpikeSorter as SS
ss=SS.SpikeSorter()

ss.run(which='spike_times',ist=2,f_s=24000,f1=300,f2=3500,order=63,path='insert-your-neural-data-recordings-path-here/C_Easy1_noise005',
            Feature_type='FSDE3',filter_type='none',training_spikes=0)

## Authors

* **Gianluca Leone**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
