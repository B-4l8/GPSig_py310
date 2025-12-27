import time
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import optimizers

def optimize(model, opt, data=None, max_iter=1000, print_freq=1, save_freq=50, val_scorer=None, history=None, callbacks=None,
                save_params=False, start_iter=0, global_step=None, var_list=None, save_best_params=False, lower_is_better=False, patience=None):
    
    # Initialize history
    if history is None or len([x for x in history.keys() if str(x).isnumeric()]) == 0:
        history = {}
        start_iter = 0
        start_time = 0.0
    else:
        # Find numeric keys to determine last iteration
        numeric_keys = [x for x in history.keys() if str(x).isnumeric()]
        start_iter = max(numeric_keys) if numeric_keys else 0
        start_time = history[start_iter]['time'] if numeric_keys else 0.0

    if 'best' not in history:
        history['best'] = {}
        history['best']['val'] = -np.inf if not lower_is_better else np.inf
        history['best']['iter'] = start_iter
        history['best']['time'] = start_time

    # Helper to get elbo/loss
    def get_elbo_val():
        if data is not None:
             return model.maximum_log_likelihood_objective(data).numpy()
        else:
             # Try parameterless, or fallback to training loss (neg likelihood)
             try:
                return model.maximum_log_likelihood_objective().numpy()
             except:
                return -model.training_loss().numpy()

    # Helper to save snapshot
    def perform_save(i, t):
        try:
            elbo = get_elbo_val()
        except:
            elbo = np.nan
            
        history[i] = {}
        history[i]['time'] = t + start_time
        history[i]['elbo'] = elbo
        
        print('Iteration {} \t|\tTime: {:.2f} \t|\tELBO: {:.2f}'.format(i, t + start_time, elbo), end='')

        if save_params:
            save_trainables = {}
            for param in model.trainable_variables:
                 save_trainables[param.name] = param.numpy()
            history[i]['params'] = save_trainables

        if callbacks is not None:
             cbs = callbacks if isinstance(callbacks, list) else [callbacks]
             history[i]['saved'] = [cb(model) for cb in cbs]

        if val_scorer is not None:
            scorers = val_scorer if isinstance(val_scorer, list) else [val_scorer]
            scores = [scr(model) for scr in scorers]
            
            for j, score in enumerate(scores):
                 label = f"Val. {j}" if len(scores) > 1 else "Val."
                 print(f" \t|\t{label}: {score:.4f}", end='')
            
            history[i]['val'] = scores if len(scores) > 1 else scores[0]
            current_score = scores[-1] 
            
            best_val = history['best'].get('val')
            if best_val is None: best_val = -np.inf if not lower_is_better else np.inf
            
            improved = (lower_is_better and current_score < best_val) or (not lower_is_better and current_score > best_val)
            
            if improved:
                history['best']['val'] = current_score
                history['best']['iter'] = i
                history['best']['time'] = t
                history['best']['elbo'] = elbo
                history['best']['params'] = {p.name: p.numpy() for p in model.trainable_variables}
            
            if patience is not None:
                if i - history['best']['iter'] > patience:
                     print('\nNo improvement for {} iterations. Stopping.'.format(patience))
                     return True 
                     
        print('') # Flush
        return False

    # Check optimizer type
    is_scipy = isinstance(opt, gpflow.optimizers.Scipy)
    
    if is_scipy:
        closure = model.training_loss_closure(data) if data is not None else model.training_loss
        print('Starting optimization with Scipy...')
        opt.minimize(closure, model.trainable_variables, options=dict(maxiter=max_iter))
        print('\nOptimization finished.')
    else:
        # TF Optimizer loop
        opts = opt if isinstance(opt, list) else [opt]
        # var_list logic
        vars_lists = [None] * len(opts)
        if var_list is not None:
             if isinstance(var_list, list) and isinstance(var_list[0], list):
                 vars_lists = var_list
             else:
                 vars_lists = [var_list]

        if data is not None:
            closure = model.training_loss_closure(data)
        else:
            closure = model.training_loss

        print('-------------------------')
        print('  Starting optimization  ')
        print('-------------------------')
        
        t0 = time.time()
        
        for i in range(start_iter + 1, start_iter + max_iter + 1):
             
             # Optimization Step
             for j, optimizer in enumerate(opts):
                 variables = vars_lists[j] if vars_lists[j] is not None else model.trainable_variables
                 with tf.GradientTape() as tape:
                     tape.watch(variables)
                     loss = closure()
                 grads = tape.gradient(loss, variables)
                 optimizer.apply_gradients(zip(grads, variables))
            
             t_now = time.time() - t0
             
             # Print / Save
             if i % print_freq == 0 or i % save_freq == 0:
                  if i % save_freq == 0:
                       stop = perform_save(i, t_now)
                       if stop: break
                  elif i % print_freq == 0:
                       try:
                           elbo = get_elbo_val()
                       except:
                           elbo = np.nan
                       print('Iteration {} \t|\tTime: {:.2f} \t|\tELBO: {:.2f}'.format(i, t_now + start_time, elbo))

        print('\nOptimization session finished...')
        
    return history