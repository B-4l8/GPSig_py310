import numpy as np
import tensorflow as tf

import gpflow
from gpflow import models, likelihoods, mean_functions
from gpflow.models import SVGP
from gpflow.kullback_leiblers import gauss_kl
from gpflow.conditionals import base_conditional
# from gpflow.conditionals import _expand_independent_outputs # This might be private or moved. 

from .inducing_variables import InducingTensors, InducingSequences, Kuu_Kuf_Kff

class SVGP(models.SVGP):
    """
    Re-implementation of SVGP from GPflow with a few minor tweaks. Slightly more efficient with SignatureKernels, and when using the low-rank option with signature kernels, this code must be used.
    """
    def __init__(self, kern, likelihood, feat, mean_function=None, num_latent_gps=None, q_diag=False, whiten=True, minibatch_size=None, num_data=None, q_mu=None, q_sqrt=None, shuffle=True, **kwargs):

        if not isinstance(feat, InducingTensors) and not isinstance(feat, InducingSequences):
            raise ValueError('feat must be of type either InducingTensors or InducingSequences')

        # In GPflow 2.x SVGP does not take X, Y. We assume data is passed to loss.
        # However, original code signature was (X, Y, kern, likelihood, feat, ...).
        # We need to adapt the signature if we want to support legacy calls, OR we assume the user will update calls.
        # Given "Migrate...", usually function signatures should stay compatible if possible.
        # But SVGP 2.x is fundamentally different regarding data storage.
        # Let's assume we change signature to match 2.x more closely OR we accept X,Y and ignore/store them?
        # The prompt says "Maintain Code Integrity: Preserve the original function arguments...".
        # So I MUST keep (X, Y, ...) in signature.
        pass

    def __init__(self, X, Y, kern, likelihood, feat, mean_function=None, num_latent=None, q_diag=False, whiten=True, minibatch_size=None, num_data=None, q_mu=None, q_sqrt=None, shuffle=True, **kwargs):
        
        # We ignore X, Y for the super().__init__ but use num_data.
        # If num_data is None, we infer from X.
        if num_data is None:
            if X is not None:
                num_data = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        # Mapping arguments
        # num_latent -> num_latent_gps
        
        super().__init__(kern, likelihood, feat, mean_function=mean_function, num_latent_gps=num_latent, q_diag=q_diag, whiten=whiten, num_data=num_data, q_mu=q_mu, q_sqrt=q_sqrt, **kwargs)
        
        self.num_data = num_data
        # We don't store X, Y in self unless necessary for some custom logic, but standard SVGP 2.x uses data passed to methods.

    
    def maximum_log_likelihood_objective(self, data):
        return self.elbo(data)

    def elbo(self, data):
        X, Y = data
        
        num_samples = tf.shape(X)[0]

        if self.whiten:
            f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
            KL =  gauss_kl(self.q_mu, tf.linalg.band_part(self.q_sqrt, -1, 0))
        else:
            f_mean, f_var, Kzz = self.predict_f(X, full_cov=False, full_output_cov=False, return_Kzz=True)
            KL =  gauss_kl(self.q_mu, tf.linalg.band_part(self.q_sqrt, -1, 0), K=Kzz)
        
        # compute variational expectations
        var_exp = self.likelihood.variational_expectations(X=X, Fmu=f_mean, Fvar=f_var, Y=Y)

        # scaling for batch size
        scale = tf.cast(self.num_data, gpflow.default_float()) / tf.cast(num_samples, gpflow.default_float())
        return tf.reduce_sum(var_exp) * scale - KL

    def predict_f(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):
        
        num_samples = tf.shape(X_new)[0]
        Kzz, Kzx, Kxx = Kuu_Kuf_Kff(self.inducing_variable, self.kernel, X_new, jitter=gpflow.config.default_jitter(), full_f_cov=full_cov)
        f_mean, f_var = base_conditional(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.linalg.band_part(self.q_sqrt, -1, 0), white=self.whiten)
        if self.mean_function is not None:
            f_mean += self.mean_function(X_new)
        
        # _expand_independent_outputs might not be needed or exposed. GPflow usually handles it.
        # But if we need consistent shape:
        # f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        
        if return_Kzz:
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var