
## Goal

We build a playground for testing methods of numerical approximation of input-output relations of neuron population models.

**Concepts**:
- *Simulation* - process of obtaining true output $y$ for input $x$.
- *Approximation/estimation/prediction* - process of obtaining an estimate output $\hat{y}$ from input $x$.
- *Model* - entity that is simulated; it has params $s$, receives inputs $x$, and produces outputs $y$.
- *IOPredictor* - entity that estimates outputs $\hat{y}$ from inputs $x$, imitating *Model*, given parameters $s$.

We will test various *IOPredictor* types on various *Model* types. We can try different predictor types for the same model family. We can play with predictor family using toy models and then test in on heavier models.

We need minimal coupling between *Model* and *IOPredictor* to have flexibility.


### Inputs/outputs

Let's rely on xarray for inputs and outputs.

Inputs and outputs are flexible in their meaning. It could be activity levels, oscillatory properties, or something else.

Some models will take time series as input and produce time-series output.
- Project mostly concerns non-temporal metrics, e.g. average over time or mean amplitude of oscillations. We should distinguish temporal inputs/outputs that Model works from non-temporal inputs/outputs that *IOPredictor* works with. Potential term for temporal entities - *signal*. Model needs metric calculator in this case.
- Some experiments may involve full time-series prediction, e.g. with autoregressive models, SINDy, etc.

In advanced cases, outputs can be distribution properties.


### Models

Models vary in complexity. Examples:
- FuncModel - explicitly computes $y=F(x)$.
- RateModel - simulates simple rate model, e.g. Wilson-Cowan.
- SpikingModel - simulates simplified spiking network, e.g. of LIF units.
- BiophysModel - detailed model simulated via NEURON.

Some models are heavy: number of simulations limited, each result precious and may deserve caching. Simulation may be deferred - e.g. submitted via Slurm on remote HPC. Workflow should be easy to restart from the same point, when results are ready.

Examples of how model can look like are in `rate_model/`.

Model parameters can be primary or derived. Derived parameters can be deterministic or random. 


### Predictors

Predictors vary in type and complexity. Examples:
- Explicit formula for regression: $\hat{y}=G(x|s)$.
- Neural network projecting $(x,s)$ onto $\hat{y}$.
- Neural density estimator: $p(\hat{y}|x,s)$

In most cases, predictors are trained to work with primary model parameters; when derived parameters are random, predictor should rely on their distributions (explicit or estimated from multiple simulation trials). In other cases, predictors work with individual random model realizations.

Predictor output can be:
- Deterministic: $\hat{y}(x,s)$
- Probabilitic: $p(\hat{y}|x,s)$

In some experiments, predictors may operate on time series.

Predictors can rely on various packages: numpy, scipy, scikit-learn, pytorch, SINDy, and others.

Examples of predictor functionality are in `experiments/resp_static/predictors` and `experiments/osc_transfer/transfer_mat.demo.py`.
