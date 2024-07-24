import jax
from flax.training import checkpoints
import numpy as np # regular old numpy
import time, datetime # measuring runtime and generating timestamps for experiments
import os, shutil # for dealing with directories
import absl  # for logging
from absl import logging # for logging
from matplotlib.colors import LogNorm

# Training utils
from src import create_train_state, train_epoch, eval_model, heatmap_grads_batches, heatmap_grads_epochs, plot_L_or_gamma

# Import configurations
# import config # Use this for the old method
from config.cli_config import config

experiment_dir = "./runs/" + config.experiment_name + "/"
if experiment_dir == "./runs/debug-test/" and os.path.isdir(experiment_dir):
    shutil.rmtree(experiment_dir)
if os.path.isdir(experiment_dir):
    raise FileExistsError("The specified directory already exists! Script stopped to prevent overwriting previous experiments. Either provide a new \"--experiment-name\" argument when calling train.py or delete the old experiment directory if you do not need it.")
os.makedirs(experiment_dir)

for experiment_index, seed in enumerate(config.seeds):
    # path to save stuff to
    timestamp = datetime.datetime.fromtimestamp(time.time())
    outpath = experiment_dir + timestamp.strftime('%Y_%m_%d_%H_%M_%S')+'/'
    os.makedirs(outpath)
    CKPT_DIR = outpath+'ckpts/'
    os.makedirs(CKPT_DIR)

    logging.use_absl_handler()
    logging.get_absl_handler().use_absl_log_file('absl_logging', outpath) 
    absl.flags.FLAGS.mark_as_parsed() 
    logging.set_verbosity(logging.INFO)

    def loginfo_and_print(msg):
        logging.info(msg)
        print(msg)
    
    loginfo_and_print(f"\n\tStarting experiment {experiment_index+1}/{len(config.seeds)}. Current seed is {seed}")
    loginfo_and_print(f"\model settings:\n{config.model}")
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    # Define model
    steps_per_epoch = len(config.train_ds['image']) // config.batch_size
    
    state = create_train_state(init_rng, config.model, config.image_dims, config.learning_rate, config.warmup_learning_rate, config.learning_rate_final, config.momentum, config.weight_decay, config.num_epochs, config.warmup_epochs, config.decay_epochs, steps_per_epoch)


    del init_rng  # Must not be used anymore.

    # Dict for storing training metrics in
    hist = {'val_loss': np.zeros(config.num_epochs), 'val_accuracy': np.zeros(config.num_epochs), 'val_top5accuracy': np.zeros(config.num_epochs), 'val_time': np.zeros(config.num_epochs),
            'train_loss': np.zeros(config.num_epochs), 'train_accuracy': np.zeros(config.num_epochs), 'train_time': np.zeros(config.num_epochs),
             'test_loss': np.nan, 'test_accuracy': np.nan, 'test_top5accuracy': np.nan, 'test_time': np.nan,
            'grad_cos_sim_batches': np.zeros((len(state.params), steps_per_epoch*config.num_epochs)),
            'grad_cos_sim_epochs': np.zeros((len(state.params), config.num_epochs)),
            'L10': np.zeros((len(state.params), config.num_epochs)),
            'L20': np.zeros((len(state.params), config.num_epochs)),
            'gamma10': np.zeros((len(state.params), config.num_epochs)),
            'gamma20': np.zeros((len(state.params), config.num_epochs))}

    best_accuracy, best_epoch = 0, 0
    epoch = 0
    for epoch in range(1, config.num_epochs+1):
        loginfo_and_print("\n====================Epoch: %d ==========================" % (epoch))

        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        # last augument turns off data augmentation for mnist
        augmentation_on = (config.dataset!="mnist") and (config.dataset!="fashionmnist")
        state, epoch_metrics, train_time = train_epoch(state, config.train_ds, config.batch_size, input_rng, augmentation_on, config.learning_algorithm, config.num_classes)
        loginfo_and_print('train: \tloss: %.4f, \taccuracy: %.4f, \truntime: %.4f' % (epoch_metrics["loss"], epoch_metrics["accuracy"], train_time))
        hist['train_loss'][epoch-1], hist['train_accuracy'][epoch-1], hist['train_time'][epoch-1] = epoch_metrics["loss"], epoch_metrics["accuracy"], train_time

        if config.learning_algorithm != "backprop":
            grad_cos_sim = np.stack(epoch_metrics["cosine_sim"]).T
            hist["grad_cos_sim_batches"][:,(epoch-1)*steps_per_epoch:(epoch)*steps_per_epoch] = grad_cos_sim
            hist["grad_cos_sim_epochs"][:,epoch-1] = grad_cos_sim.mean(axis=1)


        # Evaluate on the validation set after each training epoch 
        rng, input_rng = jax.random.split(rng)
        val_loss, val_accuracy, val_top5_accuracy, val_time, L10, L20, gamma10, gamma20 = eval_model(state, state.params, config.val_ds, config.batch_size, config.num_classes, input_rng)
        loginfo_and_print('val:  \tloss: %.4f, \taccuracy: %.4f, \ttop5_accuracy: %.4f, \truntime: %.4f' % (val_loss, val_accuracy, val_top5_accuracy, val_time))
        loginfo_and_print(f"L20: {[np.round(Li.item(), decimals=4) for Li in L20]}")
        loginfo_and_print(f"gamma20: {[np.round(gi.item(), decimals=4) for gi in gamma20]}")
        hist['val_loss'][epoch-1], hist['val_accuracy'][epoch-1], hist['val_top5accuracy'][epoch-1], hist['val_time'][epoch-1] = val_loss, val_accuracy, val_top5_accuracy, val_time
        hist['L10'][:,epoch-1] = L10
        hist['L20'][:,epoch-1] = L20
        hist['gamma10'][:,epoch-1] = gamma10
        hist['gamma20'][:,epoch-1] = gamma20
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            loginfo_and_print("Saving new checkpoint")
            checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch, keep=1, overwrite=True)
        
        if np.isnan(hist["train_loss"][epoch-1]) or np.isinf(hist["train_loss"][epoch-1]):
            loginfo_and_print("NaN or Inf encountered. Terminating training loop early")
            break
        if (epoch>5) and (hist["val_accuracy"][epoch-1] < 0.5*best_accuracy):
            loginfo_and_print("Terminating training early as validation accuracy has drastically dropped")
            break
        
    loginfo_and_print(f"\n====Loading model with best validation accuracy (epoch {best_epoch})====")
    best_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
    rng, input_rng = jax.random.split(rng)
    test_loss, test_accuracy, test_top5accuracy, test_time, _, _, _, _ = eval_model(best_state, best_state.params, config.test_ds, config.batch_size, config.num_classes, input_rng)
    hist['test_loss'], hist['test_accuracy'], hist['test_top5accuracy'], hist['test_time'] = test_loss, test_accuracy, test_top5accuracy, test_time
    loginfo_and_print('test:  \tloss: %.4f, \taccuracy: %.4f, \ttop5_accuracy: %.4f, \truntime: %.4f' % (test_loss, test_accuracy, test_top5accuracy, test_time))
    
    if config.learning_algorithm != "backprop":
        # Use color_norm=LogNorm(clip=True) for logscale plot
        heatmap_grads_epochs(hist["grad_cos_sim_epochs"], outpath+"grad_angle_epochs.pdf", True, color_norm=None)
        #Grad angle across batches in the first epoch
        first_N = 100
        heatmap_grads_batches(hist["grad_cos_sim_batches"][:,0:first_N], outpath+f"grad_angle_first_{first_N}_batches.pdf", True, color_norm=None)

    plot_L_or_gamma(hist["L20"], hist["L10"], "L", outpath+"L.pdf")
    plot_L_or_gamma(hist["gamma20"], hist["gamma10"], r"$\gamma$", outpath+"gamma.pdf")

    np.save(outpath+"hist.npy", hist)