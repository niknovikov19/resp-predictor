
1. RateModel run initialization

In a run, infer starting values for the synaptic currents from r0 (steady-state assumption).


2. Explore valid parameter combinations

Code for reference:
```
    tau_syn = np.array([taue_syn, taui_syn])
    model = RateModelWC(
        np.zeros((2, 2)), tau=0, rmax=10, gain_slope=0.5, gain_center=0)
    W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
        model, r0, tau_syn, f, tau_decay, q_ie, phi_ie
    )
    model.W = W
    model.tau_syn = tau_syn_mat
```

Create a function in a separate py-file in dev_scratch.

Arguments: model, r0 (2 values), f, ranges
ranges is a [str, np.ndarray] dict. Keys are: taue, taui, tau_decay, q_ie. Values define values to probe.
Assume tau=0 in the model.

For each combination (taue_syn, taui_syn, tau_decay, q_ie):
calculate the allowed range of phi_ie.

Return max phi_ie values as xarray with dims (taue_syn, taui_syn, tau_decay, q_ie), where.

In the main block of the file, demonstrate its work:

Make a figure with subplots. Rows: tau_decay, columns: q_ie.
Each subplot contains 2-d map of max phi_ie values. Coords of 2-d map: taue_syn, taui_syn.

Make a subfolder with the same name as py-file in dev_scratch/artifacts and save there the figure as png.

Include info to the png name: r0, f, dimensionality  (taue_syn, taui_syn, tau_decay, q_ie), and a short hash of all params.
E.g.: phi_maps_r0_5_5_f_10_dim_7_7_3_2_0eabcd12.png

Produce a json file with all info (including model params and ranges) with the same name in the same folder.


3. Test the result of make_W_h0_ei_2d_syn() with explicit noise-free simulation.

Import make_W_h0_ei_2d_syn() from rate_model/model_utils.py.
Use main_ei() from rate_model/model_utils.py as a reference.

Create a function in a new py-file in dev_scratch.

