# AMP-Discrete-Approximation
Contains script to compare AMP UQ estimates of discrete problems with their continuous approximation.

# Notes
The code is built on top of the UQPFN library as of 10th of July, 2026. As far as my knowledge goes, newer changes should not impact the computation, as it only uses uqpfn.martingale.make_alpha_schedules, uqpfn.tabpfn.forward_samples and uqpfn.tabpfn.cdf_from_samples, but one should be wary of that nonetheless. 
