#!/usr/bin/env python

import numpy as np
import yt
import argparse
import matplotlib.pyplot as plt
import pynucastro as pyna
import pandas as pd
from pynucastro.screening.screen import screen5
from yt.frontends.boxlib.api import CastroDataset
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set some fontsize settings
SMALL_SIZE = 18
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rc('xtick.major', size=7, width=2)
plt.rc('xtick.minor', size=5, width=1)
plt.rc('ytick.major', size=7, width=2)
plt.rc('ytick.minor', size=5, width=1)

def get_zone_fields(fname, r0, theta0, field_list, include_rates=False):
    """Given a fname and  radial and theta position,
    do a density-weighted average along that theta for each requested field.

    Return the max of the field_list along that theta.

    If include_rates=True, then also include 3a, c12(p,g)n13, and c12(a,g)o16 rates.

    In the future, we could return the density average normalized by density.
    """

    ds = CastroDataset(fname)
    time = ds.current_time.in_units('ms')

    ds.force_periodicity()
    dims = ds.domain_dimensions.copy()
    dims[2] = 1

    # Work with level 0 data only for now.
    cg = ds.smoothed_covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)

    r = cg["index", "r"][:, :, 0].to("km").to_ndarray()
    theta = cg["index", "theta"][:, :, 0].to_ndarray()

    r_1d = r[:, 0]
    theta_1d = theta[0, :]

    # find the 1d data that is closest to the asked r and theta
    r_idx = np.argmin(np.abs(r_1d - r0))
    theta_idx  = np.argmin(np.abs(theta_1d - theta0))

    # Find the actual field
    fields = {"t[ms]": time}
    for field in field_list:
        # Find the max of that for now.
        f = float(cg["boxlib", field][r_idx, theta_idx, 0])
        fields[field] = f

    if include_rates:
        # Evaluate 3a, c12(p,g)n13, and c12(a,g)o16

        rho = float(cg["boxlib", "density"][r_idx, theta_idx, 0])
        T = float(cg["boxlib", "Temp"][r_idx, theta_idx, 0])
        X_H1 = float(cg["boxlib", "X(H1)"][r_idx, theta_idx, 0])
        X_He4 = float(cg["boxlib", "X(He4)"][r_idx, theta_idx, 0])
        X_C12 = float(cg["boxlib", "X(C12)"][r_idx, theta_idx, 0])
        comp = pyna.Composition(["h1", "he4", "c12"])
        comp.set_nuc(pyna.Nucleus("h1"), X_H1)
        comp.set_nuc(pyna.Nucleus("he4"), X_He4)
        comp.set_nuc(pyna.Nucleus("c12"), X_C12)
        state = pyna.ThermoState(rho=rho, T=T, comp=comp)
        print(state)

        rl = pyna.ReacLibLibrary()
        rates = ["a(aa,g)c12", "c12(p,g)n13", "c12(a,g)o16"]
        for rate in rates:
            r1 = rl.get_rate_by_name(rate)
            fields[rate] = r1.eval_full_rate(state, screen_func=screen5)

    return fields


def get_weight_fields_concurrent(fnames,
                                 r0=11.028, theta0=0.15,
                                 field_list=['Temp', 'enuc', "X(H1)", "X(He4)", "X(C12)"],
                                 include_rates=True,
                                 output='profile.dat',
                                 max_workers=None):
    """
    using concurrent.futures for parallelization over reading
    different data files. Then save and return the data.
    """

    process_func = partial(get_zone_fields,
                           r0=r0, theta0=theta0,
                           field_list=field_list,
                           include_rates=include_rates)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fname = {executor.submit(process_func, fname): fname
                          for fname in fnames}

        # Collect results as they complete
        try:
            for future in as_completed(future_to_fname):
                fname = future_to_fname[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'File {fname} generated an exception: {exc}')
                    raise
        except KeyboardInterrupt:
            print(
                "\n*** got ctrl-c, cancelling remaining tasks and waiting for existing ones to finish...\n",
                flush=True,
            )
            executor.shutdown(wait=True, cancel_futures=True)
            sys.exit(1)

    # Store results as Panda Dataframe
    df = pd.DataFrame(results)

    # Sort by time
    df = df.sort_values("t[ms]").reset_index(drop=True)

    # Write to csv
    df.to_csv(output, index=False)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''This script produces
    a plot showing the density weighted average Temperature and enuc
    from a series of data files''')

    parser.add_argument('fnames', nargs='+', type=str,
                        help='''data file names for generating the plot''')
    parser.add_argument('-r', '--radial',
                        type=float, help="radial position to get zone data")
    parser.add_argument('-t', '--theta',
                        type=float, help="theta position to get zone data")
    parser.add_argument('-o', '--output', type=str, default='profile.csv',
                        help='''output name containing relevant data''')
    parser.add_argument('-j', '--jobs', default=1, type=int,
                        help="""Number of workers to plot in parallel""")
    parser.add_argument('-d', '--data', type=str,
                        help='''Data precomputed as a csv file.
                        pass in the data file here,
                        then the calculation is skipped.''')
    # parser.add_argument('-f', '--fields', nargs='+', type=str,
    #                     default=['Temp', 'enuc'],
    #                     help='''Different field names for plotting''')
    # parser.add_argument('-w', '--weighted_field', type=str, default='density',
    #                     help='''Weight different fields by this field.''')

    args = parser.parse_args()

    # Find the data points
    # Right now just find average Temp and enuc weighted by density
    # data = get_weight_fields(args.fnames, field_list=['Temp', 'enuc'],
    #                          weighted_field='density',
    #                          output=args.output,
    #                          cutoff_quantile=args.quantile)

    if args.data is None:
        data = get_weight_fields_concurrent(args.fnames,
                                            r0=args.radial, theta0=args.theta,
                                            field_list=['Temp', 'enuc', "X(H1)", "X(He4)", "X(C12)"],
                                            include_rates=True,
                                            output=args.output,
                                            max_workers=args.jobs)
    else:
        data = pd.read_csv(fname)

    # Make a time evolution plot where we have reaction rates on left y-axis
    # and mass fractions, X, on the right axis.

    time_array = data["t[ms]"]
    X_H1 = data["X(H1)"]
    X_He4 = data["X(He4)"]
    X_C12 = data["X(C12)"]

    r1 = data["a(aa,g)c12"]
    r2 = data["c12(p,g)n13"]
    r3 = data["c12(a,g)o16"]

    # Do Plotting
    fig, ax = plt.subplots()
    ax.set_xlabel('time [ms]')
    ax.plot(time_array, r1, color='tab:purple', linestyle="-", linewidth=3,
            label=r'$\alpha(\alpha \alpha, \gamma) {}^{12}\mathrm{C}$')
    ax.plot(time_array, r2, color='tab:orange', linestyle="-", linewidth=3,
            label=r'${}^{12}\mathrm{C}(p, \gamma) {}^{13}\mathrm{N}$')
    ax.plot(time_array, r3, color='tab:cyan', linestyle="-", linewidth=3,
            label=r'${}^{12}\mathrm{C}(\alpha , \gamma) {}^{12}\mathrm{C}$')
    ax.set_yscale("log")
    ax.set_ylim(1e-16, None)
    ax.set_ylabel(r'Reaction Flux [$s^{-1}$]')

    ax_twin = ax.twinx()
    ax_twin.plot(time_array, X_H1, color='tab:blue', linestyle="--", linewidth=3, label=r'$X({}^{1}\mathrm{H}$')
    ax_twin.plot(time_array, X_He4, color='tab:red', linestyle="--", linewidth=3, label=r'$X({}^{4}\mathrm{He}$')
    ax_twin.plot(time_array, X_C12, color='tab:green', linestyle="--", linewidth=3, label=r'$X({}^{12}\mathrm{C}$')
    ax_twin.set_ylabel(r'X')
    ax_twin.set_yscale("log")
    ax_twin.set_ylim([1e-10, 1.1])
    # Combine legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='lower right', frameon=False)

    fig.tight_layout()
    fig.set_size_inches(8, 8)
    fig.savefig("rate_profile.png", bbox_inches="tight")
