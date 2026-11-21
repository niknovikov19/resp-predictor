
## Goal

We build a playground for testing methods of numerical approximation of input-output relations of neuron population models.


Concepts:
- Simulation - process of obtaining true output y for input x.
- Approximation/estimation/prediction - process of obtaining an estimate output yhat from input x.
- Model - entity that is simulated; it has params, receives inputs, and produces outputs.
- IOPredictor - entity that estimates outputs from inputs, imitating Model.

We will test various IOPredictor types on various Model types. We can try different predictor types for the same model family. We can play with predictor family using toy models and then test in on heavier models.

We need little coupling between Model and IOPredictor. 

### Inputs/outputs

Let's rely on xarray for inputs and outputs.

Inputs and outputs are flexible in their meaning. It could be activity levels, oscillatory properties, or something else.

Some models will take time series as input and produce time-series output. Project mostly concerns non-temporal metrics, e.g. average over time or mean amplitude of oscillations. We should distinguish temporal inputs/outputs that Model works from non-temporal inputs/outputs that IOPredictor works with. Potential term for temporal entities - signal. Model needs metric calculator.

### Models

Models vary in complexity. Examples:
- FuncModel - explicitly computes y=F(x).
- RateModel - simulates simple rate model, e.g. Wilson-Cowan.
- SpikingModel - simulates simplified spiking network, e.g. of LIF units.
- BiophysModel - detailed model simulated via NEURON.

Some models are heavy: number of simulations limited, each result precious. Simulation may be deferred - e.g. submitted via Slurm on remote HPC. Workflow should be easy to restart from the same point, when results are ready.

In resp_predictor repo, `rate_model/` 





